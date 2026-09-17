#![allow(dead_code)]
use std::fmt;

use burn::tensor::{
    ElementConversion, Tensor,
    activation::{log_softmax, softmax},
    backend::Backend,
    loss::cross_entropy_with_logits,
};

#[derive(Debug)]
pub struct LossStats<B: Backend> {
    pub policy: Tensor<B, 1>,
    pub value: Tensor<B, 1>,
    pub total: Tensor<B, 1>,
}

#[derive(Debug, Clone, Copy)]
pub struct LossValue {
    pub policy: f32,
    pub value: f32,
    pub total: f32,
    pub target_entropy: Option<f32>,
    pub pred_entropy: Option<f32>,
}

impl LossValue {
    /// KL = policy_ce - target_entropy
    pub fn kl(&self) -> Option<f32> {
        self.target_entropy.map(|h| (self.policy - h).max(0.0))
    }
}

fn fmt_opt(v: Option<f32>) -> String {
    match v {
        Some(x) => format!("{:>7.4}", x),
        None => "   --  ".to_string(),
    }
}

impl fmt::Display for LossValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "policy={:>7.4} value={:>7.4} total={:>7.4} | H(π)={} H(p)={} KL={}",
            self.policy,
            self.value,
            self.total,
            fmt_opt(self.target_entropy),
            fmt_opt(self.pred_entropy),
            fmt_opt(self.kl()),
        )
    }
}

pub fn target_entropy<B: Backend>(target: Tensor<B, 2>) -> f32 {
    let log_target = target.clone().clamp_min(1e-12).log();
    let h = -(target * log_target).sum_dim(1);
    h.mean().into_scalar().elem::<f32>()
}

pub fn pred_entropy<B: Backend>(logits: Tensor<B, 2>) -> f32 {
    let p = softmax(logits.clone(), 1);
    let log_p = log_softmax(logits, 1);
    let h = -(p * log_p).sum_dim(1);
    h.mean().into_scalar().elem::<f32>()
}

pub fn policy_loss<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    cross_entropy_with_logits(logits, target)
}

pub fn value_loss<B: Backend>(prediction: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 2> {
    (prediction - target).powf_scalar(2.0)
}

pub fn total_loss<B: Backend>(
    policy_logits: Tensor<B, 2>,
    target_policy: Tensor<B, 2>,
    value_prediction: Tensor<B, 2>,
    target_value: Tensor<B, 2>,
) -> LossStats<B> {
    let policy = policy_loss(policy_logits, target_policy).mean();
    let value = value_loss(value_prediction, target_value).mean();

    LossStats {
        total: policy.clone() + value.clone(),
        policy,
        value,
    }
}
