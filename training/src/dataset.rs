#![allow(dead_code)]

pub struct TrainingSample {
    pub state: Vec<f32>,
    pub policy: Vec<f32>,
    pub value: f32,
}
