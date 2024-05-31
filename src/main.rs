#![no_std]
#![no_main]

use embassy_executor::Spawner;
use nalgebra::*;
use weights::{HIDDEN_WEIGHTS, INPUT_EXAMPLE, INPUT_WEIGHTS};
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

fn softmax(input: &[f32; 10]) -> [f32; 10] {
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

    softmax_values
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + custom_exp(-x))
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let _p = embassy_rp::init(Default::default());
    defmt::info!("starting crong");

    let input_weights: SMatrix<f32, 784, 10> =
        SMatrix::from_row_iterator(INPUT_WEIGHTS.iter().cloned());
    let hidden_biases: SMatrix<f32, 1, 10> = SMatrix::from_row_iterator(
        [
            -0.16926917,
            -0.09152439,
            0.46151575,
            -0.27533257,
            -1.2676352,
            0.0024831702,
            1.3180981,
            -0.07180291,
            -0.6612474,
            1.2763305,
        ]
        .iter()
        .cloned(),
    );
    let output_weights: SMatrix<f32, 10, 10> =
        SMatrix::from_row_iterator(HIDDEN_WEIGHTS.iter().cloned());
    let output_biases: SMatrix<f32, 1, 10> = SMatrix::from_row_iterator(
        [
            -0.11156949,
            0.8989733,
            0.7401263,
            -0.27717194,
            -0.19641854,
            -0.23445761,
            -1.0752414,
            0.32264754,
            0.05256069,
            -0.27262202,
        ]
        .iter()
        .cloned(),
    );

    let input: SMatrix<f32, 1, 784> = SMatrix::from_row_iterator(INPUT_EXAMPLE.iter().cloned());

    let mut nn = input * input_weights;
    nn += hidden_biases;

    // Sigmoid
    for i in 0..nn.len() {
        nn[i] = sigmoid(nn[i]);
    }

    nn *= output_weights;
    nn += output_biases;

    // Softmax
    // Convert to &[f32; 10] array to use the softmax function
    let nn = nn.as_slice();
    // make it &[f32; 10]
    let nn: &[f32; 10] = nn.try_into().unwrap(); // safe because we know the length is 10
    let nn = softmax(nn);

    let len = nn.len();
    for i in 0..len {
        defmt::info!("{}: {}", i, nn[i]);
    }
}
