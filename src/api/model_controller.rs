use lightray_core::lightray_torch::TorchScriptGraph;
use std::fs;
use std::io::Write;

use actix_multipart::Multipart;
use actix_web::{error::BlockingError, web, Error, HttpResponse};
use futures::StreamExt;
use tch::CModule;
use uuid::Uuid;

use lightray_core::lightray_executor::{
    LightrayIValueSemantic, LightrayModel, LightrayModelId, LightrayModelSemantics,
};

use lightray_core::lightray_executor::executor::{
    InMemorySimpleLightrayExecutor, LightrayExecutor,
};

use lightray_core::lightray_executor::errors::LightrayRegistrationError;

use lightray_core::lightray_torch::TorchScriptInput;

use crate::api::errors::ServiceError;

pub async fn upload_model(
    executor: web::Data<InMemorySimpleLightrayExecutor>,
    mut c_module: Multipart,
) -> Result<HttpResponse, Error> {
    fs::create_dir_all("./model_store")?;
    let mut filepath: Option<String> = None;

    while let Some(item) = c_module.next().await {
        let mut field = item?;
        let content_type = field.content_disposition().unwrap();
        let filename = content_type.get_filename().unwrap();
        let create_filepath = format!("./model_store/{}", filename);
        filepath = Some(create_filepath.clone());

        let mut f = web::block(|| std::fs::File::create(create_filepath))
            .await
            .unwrap();
        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            f = web::block(move || f.write_all(&data).map(|_| f)).await?;
        }
    }

    if let Some(file) = filepath {
        register_model(file, executor).await
    } else {
        Err(Into::<ServiceError>::into(LightrayRegistrationError::MissingModel).into())
    }
}

pub async fn delete_model(
    executor: web::Data<InMemorySimpleLightrayExecutor>,
    params: web::Path<LightrayModelId>,
) -> Result<HttpResponse, ServiceError> {
    let model_id = LightrayModelId {
        model_id: params.model_id,
        model_version: params.model_version,
    };

    match web::block(move || executor.delete_model(model_id)).await {
        Ok(()) => Ok(HttpResponse::Ok().finish()),
        Err(err) => match err {
            BlockingError::Canceled => Err(ServiceError::InternalServerError),
            BlockingError::Error(lightray_reg_err) => Err(lightray_reg_err.into()),
        },
    }
}

pub async fn execute_model(
    executor: web::Data<InMemorySimpleLightrayExecutor>,
    params: web::Path<LightrayModelId>,
    input: web::Json<TorchScriptInput>,
) -> Result<HttpResponse, ServiceError> {
    let model_id = LightrayModelId {
        model_id: params.model_id,
        model_version: params.model_version,
    };

    match web::block(move || executor.execute(&model_id, &input, false)).await {
        Ok(stats) => Ok(HttpResponse::Ok().json(stats)),
        Err(err) => match err {
            BlockingError::Canceled => Err(ServiceError::InternalServerError),
            BlockingError::Error(lightray_exec_err) => Err(lightray_exec_err.into()),
        },
    }
}

async fn register_model(
    file: String,
    executor: web::Data<InMemorySimpleLightrayExecutor>,
) -> Result<HttpResponse, Error> {
    let graph = TorchScriptGraph {
        batchable: false,
        module: CModule::load(file).unwrap(),
    };
    let lightray_id = LightrayModelId {
        model_id: Uuid::new_v4(),
        model_version: 0,
    };

    let lightray_model = LightrayModel::new(
        lightray_id,
        graph,
        vec![],
        LightrayModelSemantics {
            positional_semantics: vec![
                LightrayIValueSemantic::TypeMatch,
                LightrayIValueSemantic::ExactValueMatch,
                LightrayIValueSemantic::ExactValueMatch,
            ],
        },
    )
    .unwrap();

    match web::block(move || executor.register_model(lightray_model)).await {
        Ok(model_id) => Ok(HttpResponse::Ok().json(model_id)),
        Err(err) => match err {
            BlockingError::Canceled => Err(ServiceError::InternalServerError.into()),
            BlockingError::Error(lightray_reg_err) => {
                Err(Into::<ServiceError>::into(lightray_reg_err).into())
            }
        },
    }
}
