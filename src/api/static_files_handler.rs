use actix_web::HttpResponse;

pub fn index() -> HttpResponse {
    let html = r#"<html>
        <head><title>Upload Test</title></head>
        <body>
            <form action="/api/model" method="post" enctype="multipart/form-data">
                Model file:<br>
                <input type="file" name="model_file"/><br>
                Model samples (json format):<br>
                <textarea name="samples" cols="40" rows="5">[{"positional_arguments":[{"List":[{"Str":"<bos>"},{"Str":"call"},{"Str":"mom"},{"Str":"<eos>"}]},{"Int":3},{"Int":3}]}]</textarea><br>
                Model semantics (json format):<br>
                <textarea name="semantics" cols="40" rows="5">{"positional_semantics":["TypeMatch","ExactValueMatch","ExactValueMatch"]}</textarea><br>
                <input type="submit" value="Submit"></button>
            </form>
        </body>
    </html>"#;

    HttpResponse::Ok().body(html)
}
