#![no_std]
#![no_main]

use embassy_executor::Spawner;
use nalgebra::*;
use num_traits::float::FloatCore;
use weights::{HIDDEN_BIASES, HIDDEN_WEIGHTS, INPUT_BIASES, INPUT_EXAMPLE, INPUT_WEIGHTS};
use {defmt_rtt as _, panic_probe as _};

mod weights;

fn custom_exp(x: f32) -> f32 {
    // This is a custom implementation of the exponential function that uses a Taylor series
    const TERMS: usize = 20;
    let mut sum = 1.0;
    let mut term = 1.0;

    for n in 1..TERMS {
        term *= x / n as f32;
        sum += term;
    }

    sum
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + custom_exp(-x))
}

fn softmax(input: &SMatrix<f32, 1, 10>) -> SMatrix<f32, 1, 10> {
    let mut exp_values = [0.0; 10];
    let mut sum_exp = 0.0;

    // Calculate the exponential of each input value and the sum of all exponentials
    for i in 0..10 {
        exp_values[i] = custom_exp(input[i]);
        sum_exp += exp_values[i];
    }

    // Calculate the softmax values
    let mut softmax_values = [0.0; 10];
    for i in 0..10 {
        softmax_values[i] = exp_values[i] / sum_exp;
    }

    softmax_values.into()
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let _p = embassy_rp::init(Default::default());
    defmt::info!("Starting NN");

    let now = embassy_time::Instant::now();
    let input_weights: SMatrix<f32, 784, 10> =
        SMatrix::from_row_iterator(INPUT_WEIGHTS.iter().cloned());
    let input_biases: SMatrix<f32, 1, 10> =
        SMatrix::from_row_iterator(INPUT_BIASES.iter().cloned());
    let output_weights: SMatrix<f32, 10, 10> =
        SMatrix::from_row_iterator(HIDDEN_WEIGHTS.iter().cloned());
    let output_biases: SMatrix<f32, 1, 10> =
        SMatrix::from_row_iterator(HIDDEN_BIASES.iter().cloned());
    let loaded_weights = embassy_time::Instant::now();

    let input: SMatrix<f32, 1, 784> = SMatrix::from_row_iterator(INPUT_EXAMPLE.iter().cloned());

    let mut nn = input * input_weights;
    nn += input_biases;

    // Sigmoid
    for i in 0..nn.len() {
        nn[i] = sigmoid(nn[i]);
    }

    nn *= output_weights;
    nn += output_biases;

    // Softmax
    let nn = softmax(&nn);

    let end = embassy_time::Instant::now();

    defmt::info!(
        "Loaded weights in {:?}ms",
        (loaded_weights - now).as_millis()
    );
    defmt::info!("Ran NN in {:?}ms", (end - loaded_weights).as_millis());
    defmt::info!("Total time: {:?}ms", (end - now).as_millis());
    defmt::info!(
        "You could run this {} times per second",
        1000 / (end - now).as_millis()
    );

    let len = nn.len();
    for i in 0..len {
        let rounded_value = (nn[i] * 10000.0).round() / 100.0;
        defmt::info!("{}: {}%", i, rounded_value);
    }
}
