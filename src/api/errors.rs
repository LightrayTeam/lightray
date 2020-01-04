use actix_web::{error::ResponseError, HttpResponse};
use derive_more::Display;

use lightray_core::lightray_executor::errors::{
    LightrayModelExecutionError, LightrayModelInputSemanticError, LightrayModelVerificationError,
    LightrayRegistrationError,
};

#[derive(Debug, Display)]
pub enum ServiceError {
    #[display(fmt = "Internal Server Error")]
    InternalServerError,

    #[display(fmt = "BadRequest: {}", _0)]
    BadRequest(String),

    #[display(fmt = "Unauthorized")]
    Unauthorized,
}

impl ResponseError for ServiceError {
    fn error_response(&self) -> HttpResponse {
        match self {
            ServiceError::InternalServerError => {
                HttpResponse::InternalServerError().json("Internal Server Error, Please try later")
            }
            ServiceError::BadRequest(ref message) => HttpResponse::BadRequest().json(message),
            ServiceError::Unauthorized => HttpResponse::Unauthorized().json("Unauthorized"),
        }
    }
}

impl From<LightrayRegistrationError> for ServiceError {
    fn from(error: LightrayRegistrationError) -> ServiceError {
        match error {
            LightrayRegistrationError::LightrayModelVerificationError(model_err) => match model_err
            {
                LightrayModelVerificationError::InternalTorchError(torch_err) => {
                    ServiceError::BadRequest(torch_err.internal_error)
                }
                LightrayModelVerificationError::LightrayMissingSamples(_) => {
                    ServiceError::BadRequest(String::from("Missing samples"))
                }
            },
            _ => ServiceError::InternalServerError,
        }
    }
}

impl From<LightrayModelExecutionError> for ServiceError {
    fn from(error: LightrayModelExecutionError) -> ServiceError {
        match error {
            LightrayModelExecutionError::LightrayModelInputSemanticError(model_err) => {
                match model_err {
                    LightrayModelInputSemanticError::LightrayVerificationInputSize(err) => {
                        ServiceError::BadRequest(err.to_string())
                    }
                    LightrayModelInputSemanticError::LightrayVerificationInputTypes(err) => {
                        ServiceError::BadRequest(err.to_string())
                    }
                    LightrayModelInputSemanticError::LightrayVerificationInputDoesNotEqual(err) => {
                        ServiceError::BadRequest(err.to_string())
                    }
                    LightrayModelInputSemanticError::LightrayVerificationInputSizeDoesNotEqual(
                        err,
                    ) => ServiceError::BadRequest(err.to_string()),
                }
            }
            _ => ServiceError::InternalServerError,
        }
    }
}
