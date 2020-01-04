use actix_web::{web, App, HttpServer};

use lightray::api::{model_controller, static_files_handler};
use lightray_core::lightray_executor::executor::InMemorySimpleLightrayExecutor;
use lightray_core::lightray_scheduler::greedy_fifo_queue::LightrayFIFOWorkQueue;

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var(
        "RUST_LOG",
        "lightray=debug,actix_web=info,actix_server=info",
    );
    HttpServer::new(move || {
        App::new()
            .data(InMemorySimpleLightrayExecutor::new())
            .service(web::resource("/").route(web::get().to(static_files_handler::index)))
            .service(
                web::scope("/api")
                    .service(
                        web::resource("/model")
                            .route(web::post().to(model_controller::upload_model)),
                    )
                    .service(
                        web::resource("/model/{model_id}/version/{model_version}")
                            .route(web::delete().to(model_controller::delete_model)),
                    )
                    .service(
                        web::resource("/model/{model_id}/version/{model_version}")
                            .route(web::post().to(model_controller::execute_model)),
                    ),
            )
    })
    .bind("127.0.0.1:5000")?
    .run()
    .await
}
