use actix_web::{web, App, HttpServer, Responder};
use listenfd::ListenFd;

fn index() -> impl Responder {
    "Hello World!"
}


fn main() {
    let mut listenfd = ListenFd::from_env();
    let mut server = HttpServer::new(|| App::new().route("/", web::get().to(index)));

    server = if let Some(l) = listenfd.take_tcp_listener(0).unwrap() {
        server.listen(l).unwrap()
    } else {
        server.bind("127.0.0.1:5000").unwrap()
    };

    server.run().unwrap();
}