use log::{
    set_logger, set_max_level, Level, LevelFilter, Log, Metadata, Record,
};

struct Logger;
static GLOBAL_LOG: Logger = Logger;

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("% [{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

pub fn start_logging() {
    let _ = set_logger(&GLOBAL_LOG);
    set_max_level(LevelFilter::Info);
}
