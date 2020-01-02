//! Numpy support for tensors.
//!
//! Format spec:
//! https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html
use std::collections::HashMap;
use std::io::{BufReader, Read};
use tch::{Kind, Tensor};

const NPY_MAGIC_STRING: &[u8] = b"\x93NUMPY";

fn read_header<R: Read>(buf_reader: &mut BufReader<R>) -> Result<String, String> {
    let mut magic_string = vec![0u8; NPY_MAGIC_STRING.len()];
    if let Err(x) = buf_reader.read_exact(&mut magic_string) {
        return Err(x.to_string());
    }
    if magic_string != NPY_MAGIC_STRING {
        return Err("magic string mismatch".to_string());
    };
    let mut version = [0u8; 2];
    if let Err(error) = buf_reader.read_exact(&mut version) {
        return Err(error.to_string());
    }
    let header_len_len = match version[0] {
        1 => 2,
        2 => 4,
        otherwise => return Err(format!("unsupported version {}", otherwise)),
    };
    let mut header_len = vec![0u8; header_len_len];
    if let Err(error) = buf_reader.read_exact(&mut header_len) {
        return Err(error.to_string());
    }
    let header_len = header_len
        .iter()
        .rev()
        .fold(0 as usize, |acc, &v| 256 * acc + v as usize);
    let mut header = vec![0u8; header_len];
    if let Err(error) = buf_reader.read_exact(&mut header) {
        return Err(error.to_string());
    }
    Ok(String::from_utf8_lossy(&header).to_string())
}

#[derive(Debug, PartialEq)]
struct Header {
    descr: Kind,
    fortran_order: bool,
    shape: Vec<i64>,
}

impl Header {
    fn parse(header: &str) -> Result<Header, String> {
        let header =
            header.trim_matches(|c: char| c == '{' || c == '}' || c == ',' || c.is_whitespace());

        let mut parts: Vec<String> = vec![];
        let mut start_index = 0usize;
        let mut cnt_parenthesis = 0i64;
        for (index, c) in header.chars().enumerate() {
            match c {
                '(' => cnt_parenthesis += 1,
                ')' => cnt_parenthesis -= 1,
                ',' => {
                    if cnt_parenthesis == 0 {
                        parts.push(header[start_index..index].to_owned());
                        start_index = index + 1;
                    }
                }
                _ => {}
            }
        }
        parts.push(header[start_index..].to_owned());
        let mut part_map: HashMap<String, String> = HashMap::new();
        for part in parts.iter() {
            let part = part.trim();
            if !part.is_empty() {
                match part.split(':').collect::<Vec<_>>().as_slice() {
                    [key, value] => {
                        let key = key.trim_matches(|c: char| c == '\'' || c.is_whitespace());
                        let value = value.trim_matches(|c: char| c == '\'' || c.is_whitespace());
                        let _ = part_map.insert(key.to_owned(), value.to_owned());
                    }
                    _ => return Err(format!("unable to parse header {}", header)),
                }
            }
        }
        let fortran_order = match part_map.get("fortran_order") {
            None => false,
            Some(fortran_order) => match fortran_order.as_ref() {
                "False" => false,
                "True" => true,
                _ => return Err(format!("unknown fortran_order {}", fortran_order)),
            },
        };
        let descr = match part_map.get("descr") {
            None => return Err(format!("no descr in header")),
            Some(descr) => {
                if descr.is_empty() {
                    return Err("empty descr".to_string());
                }
                if descr.starts_with('>') {
                    return Err(format!("little-endian descr {}", descr));
                }
                match descr.trim_matches(|c: char| c == '=' || c == '<') {
                    "f4" => Kind::Float,
                    "f8" => Kind::Double,
                    "i4" => Kind::Int,
                    "i8" => Kind::Int64,
                    "i2" => Kind::Int16,
                    "i1" => Kind::Int8,
                    "u1" => Kind::Uint8,
                    descr => return Err(format!("unrecognized descr {}", descr)),
                }
            }
        };
        let shape = match part_map.get("shape") {
            None => return Err("no shape in header".to_string()),
            Some(shape) => {
                let shape = shape.trim_matches(|c: char| c == '(' || c == ')' || c == ',');
                if shape.is_empty() {
                    vec![]
                } else {
                    if let Err(error) = shape
                        .split(',')
                        .map(|v| v.trim().parse::<i64>())
                        .collect::<Result<Vec<_>, _>>()
                    {
                        return Err(error.to_string());
                    }
                    vec![]
                }
            }
        };
        Ok(Header {
            descr,
            fortran_order,
            shape,
        })
    }
}

pub fn read_npy(value: &String) -> Result<Tensor, String> {
    let mut buf_reader = BufReader::new(value.as_bytes());
    let header = read_header(&mut buf_reader)?;
    let header = Header::parse(&header)?;
    if header.fortran_order {
        return Err("fortran order not supported".to_string());
    }
    let mut data: Vec<u8> = vec![];
    if let Err(error) = buf_reader.read_to_end(&mut data) {
        return Err(error.to_string());
    }
    match Tensor::f_of_data_size(&data, &header.shape, header.descr) {
        Result::Ok(tensor) => Ok(tensor),
        Result::Err(x) => Err(x.to_string()),
    }
}
