use lightray_core::lightray_torch::TorchScriptGraph;
use std::fs;
use std::io::Write;

use actix_multipart::{Field, Multipart};
use actix_web::{error::BlockingError, web, Error, HttpResponse};
use futures::StreamExt;
use tch::CModule;
use uuid::Uuid;

use lightray_core::lightray_executor::{LightrayModel, LightrayModelId, LightrayModelSemantics};

use lightray_core::lightray_executor::executor::{
    InMemorySimpleLightrayExecutor, LightrayExecutor,
};

use lightray_core::lightray_scheduler::greedy_fifo_queue::LightrayFIFOWorkQueue;
use lightray_core::lightray_scheduler::queue::LightrayWorkQueue;
use lightray_core::lightray_torch::TorchScriptInput;

use crate::api::errors::ServiceError;
use crate::api::multipart_utils::read_multipart_json;

pub async fn upload_model(
    queue: web::Data<LightrayFIFOWorkQueue<InMemorySimpleLightrayExecutor>>,
    mut c_module: Multipart,
) -> Result<HttpResponse, Error> {
    fs::create_dir_all("./model_store")?;
    let mut filepath: Option<String> = None;
    let mut samples: Option<Vec<TorchScriptInput>> = None;
    let mut semantics: Option<LightrayModelSemantics> = None;

    while let Some(item) = c_module.next().await {
        let mut field = item?;
        let content_type = field.content_disposition().unwrap();

        match content_type.get_name() {
            Some("model_file") => {
                filepath = Some(save_model_file(&mut field, content_type.get_filename()).await?);
            }
            Some("samples") => {
                samples = Some(get_samples(&mut field).await?);
            }
            Some("semantics") => {
                semantics = Some(get_model_semantics(&mut field).await?);
            }
            Some(other) => {
                return Err(ServiceError::BadRequest(format!(
                    "unsupported formdata field: {}",
                    other
                ))
                .into())
            }
            None => {
                return Err(ServiceError::BadRequest("unspecified formdata field".to_string()).into())
            }
        }
    }

    register_model(filepath, samples, semantics, queue).await
}

pub async fn delete_model(
    queue: web::Data<LightrayFIFOWorkQueue<InMemorySimpleLightrayExecutor>>,
    params: web::Path<LightrayModelId>,
) -> Result<HttpResponse, ServiceError> {
    let model_id = LightrayModelId {
        model_id: params.model_id,
        model_version: params.model_version,
    };

    match web::block(move || queue.get_executor().delete_model(model_id)).await {
        Ok(()) => Ok(HttpResponse::Ok().finish()),
        Err(err) => match err {
            BlockingError::Canceled => Err(ServiceError::InternalServerError),
            BlockingError::Error(lightray_reg_err) => Err(lightray_reg_err.into()),
        },
    }
}

pub async fn execute_model(
    queue: web::Data<LightrayFIFOWorkQueue<InMemorySimpleLightrayExecutor>>,
    params: web::Path<LightrayModelId>,
    input: web::Json<TorchScriptInput>,
) -> Result<HttpResponse, ServiceError> {
    let model_id = LightrayModelId {
        model_id: params.model_id,
        model_version: params.model_version,
    };

    match web::block(move || queue.get_executor().execute(&model_id, &input, false)).await {
        Ok(stats) => Ok(HttpResponse::Ok().json(stats)),
        Err(err) => match err {
            BlockingError::Canceled => Err(ServiceError::InternalServerError),
            BlockingError::Error(lightray_exec_err) => Err(lightray_exec_err.into()),
        },
    }
}

async fn get_samples(mut field: &mut Field) -> Result<Vec<TorchScriptInput>, Error> {
    match read_multipart_json::<Vec<TorchScriptInput>>(
        &mut field,
    )
    .await {
        Ok(s) => {
            Ok(s)
        }
        Err(json_error) => {
            return Err(
                ServiceError::BadRequest(format!("Model samples JSON deserialize error: {}", json_error)).into(),
            );
        }
    }
}

async fn get_model_semantics(mut field: &mut Field) -> Result<LightrayModelSemantics, Error> {
    match read_multipart_json::<LightrayModelSemantics>(
        &mut field,
    )
    .await {
        Ok(s) => {
            Ok(s)
        }
        Err(json_error) => {
            return Err(
                ServiceError::BadRequest(format!("Model semantics JSON deserialize error: {}", json_error)).into(),
            );
        }
    }
}

async fn save_model_file(field: &mut Field, filename: Option<&str>) -> Result<String, Error> {
    let filepath: String;
    match filename {
        Some("") => {
            return Err(
                ServiceError::BadRequest("no filename provided".to_string()).into()
            );
        }
        Some(file) => {
            filepath = format!("./model_store/{}", file);
        }
        None => {
            return Err(
                ServiceError::BadRequest("no filename provided".to_string()).into()
            );
        }
    }
    let create_filepath = filepath.clone();
    let mut f = web::block(|| std::fs::File::create(create_filepath))
        .await
        .unwrap();
    while let Some(chunk) = field.next().await {
        let data = chunk.unwrap();
        f = web::block(move || f.write_all(&data).map(|_| f)).await?;
    }
    Ok(filepath)
}

async fn register_model(
    file: Option<String>,
    samples: Option<Vec<TorchScriptInput>>,
    semantics: Option<LightrayModelSemantics>,
    queue: web::Data<LightrayFIFOWorkQueue<InMemorySimpleLightrayExecutor>>,
) -> Result<HttpResponse, Error> {
    let input_file = file.ok_or_else(||
        Into::<Error>::into(ServiceError::BadRequest(String::from("missing TorchScript file"))),
    )?;
    let input_samples = samples
        .ok_or_else(|| Into::<Error>::into(ServiceError::BadRequest(String::from("missing input samples"))))?;
    let input_semantics = semantics
        .ok_or_else(|| Into::<Error>::into(ServiceError::BadRequest(String::from("missing model semantics"))))?;

    let graph = TorchScriptGraph {
        batchable: false,
        module: CModule::load(input_file).unwrap(),
    };
    let lightray_id = LightrayModelId {
        model_id: Uuid::new_v4(),
        model_version: 0,
    };

    let lightray_model =
        LightrayModel::new(lightray_id, graph, input_samples, input_semantics).unwrap();

    match web::block(move || queue.get_executor().register_model(lightray_model)).await {
        Ok(model_id) => Ok(HttpResponse::Ok().json(model_id)),
        Err(err) => match err {
            BlockingError::Canceled => Err(ServiceError::InternalServerError.into()),
            BlockingError::Error(lightray_reg_err) => {
                Err(Into::<ServiceError>::into(lightray_reg_err).into())
            }
        },
    }
}
