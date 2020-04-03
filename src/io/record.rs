pub trait Record {
    fn start_inference(&mut self, _inference: &'static str) {}
    fn end_inference(&mut self) {}
}

pub struct Silent;
impl Record for Silent {}
