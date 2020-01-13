use actix_multipart::Field;
use actix_web::Error;
use bytes::{Bytes, BytesMut};
use futures::stream::StreamExt;
use serde::de::DeserializeOwned;

use crate::api::errors::ServiceError;

const JSON_FIELD_CAPACITY: usize = 1_000_000;

pub async fn read_multipart_data(field: &mut Field) -> Result<Bytes, Error> {
    let mut b = BytesMut::with_capacity(JSON_FIELD_CAPACITY);
    loop {
        match field.next().await {
            Some(Ok(chunk)) => {
                if (b.len() + chunk.len()) <= JSON_FIELD_CAPACITY {
                    b.extend_from_slice(&chunk)
                } else {
                    return Err(ServiceError::BadRequest(String::from("payload too large")).into());
                }
            }
            None => return Ok(b.freeze()),
            _ => return Err(ServiceError::InternalServerError.into()),
        }
    }
}

pub async fn read_multipart_file(field: &mut Field) -> Result<Bytes, Error> {
    let mut b = BytesMut::new();
    loop {
        match field.next().await {
            Some(Ok(chunk)) => b.extend_from_slice(&chunk),
            None => return Ok(b.freeze()),
            _ => return Err(ServiceError::InternalServerError.into()),
        }
    }
}

pub async fn read_multipart_json<T: DeserializeOwned>(mut field: &mut Field) -> Result<T, Error> {
    let data = read_multipart_data(&mut field).await?;
    Ok(serde_json::from_slice::<T>(&data)?)
}
