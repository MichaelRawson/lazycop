pub trait Record {}

pub struct NoProof;
impl Record for NoProof {}

pub struct Proof;
impl Record for Proof {}
