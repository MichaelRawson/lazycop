use log::LevelFilter;

pub fn start_logging() {
    simple_logging::log_to_stderr(LevelFilter::Info);
}
