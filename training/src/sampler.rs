use hex_go::game::action::Action;
use rand::RngExt;

pub fn sample_action_by_temperature(policy: &[(Action, f32)], temperature: f32) -> Action {
    assert!(!policy.is_empty());
    assert!(temperature >= 0.0);

    if temperature <= 1e-4 {
        return policy.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    }

    let max_probability = policy
        .iter()
        .map(|(_, probability)| *probability)
        .fold(0.0_f32, f32::max);

    if max_probability <= 0.0 {
        return policy.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    }

    let exponent = 1.0 / temperature;

    let weights: Vec<f32> = policy
        .iter()
        .map(|(_, probability)| (probability / max_probability).powf(exponent))
        .collect();

    let total: f32 = weights.iter().sum();

    let mut target = rand::rng().random::<f32>() * total;

    for ((action, _), weight) in policy.iter().zip(&weights) {
        target -= weight;

        if target <= 0.0 {
            return *action;
        }
    }

    policy.last().unwrap().0
}
