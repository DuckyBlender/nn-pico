#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_time::{Duration, Timer};
use nalgebra::*;
use {defmt_rtt as _, panic_probe as _};

struct XorshiftRng {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl XorshiftRng {
    fn next_u32(&mut self) -> u32 {
        let t = self.x ^ (self.x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        self.w = self.w ^ (self.w >> 19) ^ (t ^ (t >> 8));
        self.w
    }
}

static mut XORSHIFT_RNG: XorshiftRng = XorshiftRng {
    x: 123456789,
    y: 362436069,
    z: 521288629,
    w: 88675123,
};

fn random_f32() -> f32 {
    unsafe {
        let random_u32 = XORSHIFT_RNG.next_u32();
        let random_f32 = (random_u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        random_f32
    }
}

//write me sigmoid function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x) * (-x))
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let _p = embassy_rp::init(Default::default());
    // Add two together
    defmt::info!("starting crong");

    let mut input: Matrix<f32, Const<1>, Const<784>, ArrayStorage<f32, 1, 784>> = Matrix::zeros();
    let mut hidden_weights: Matrix<f32, Const<784>, Const<10>, ArrayStorage<f32, 784, 10>> =
        Matrix::zeros();
    let mut hidden_biases: Matrix<f32, Const<1>, Const<10>, ArrayStorage<f32, 1, 10>> = Matrix::zeros();
    let mut output_weights: Matrix<f32, Const<10>, Const<10>, ArrayStorage<f32, 10, 10>> =
        Matrix::zeros();
    let mut output_biases: Matrix<f32, Const<1>, Const<10>, ArrayStorage<f32, 1, 10>> = Matrix::zeros();

    // randomize all the weights and biases
    
    
    
    loop {
        for i in 0..input.len() {
            input[i] = random_f32();
        }    
    
        for i in 0..hidden_weights.len() {
            hidden_weights[i] = random_f32();
        }
    
        for i in 0..hidden_biases.len() {
            hidden_biases[i] = random_f32();
        }
    
        for i in 0..output_weights.len() {
            output_weights[i] = random_f32();
        }
    
        for i in 0..output_biases.len() {
            output_biases[i] = random_f32();
        }

        let mut test = input * hidden_weights;
        test += hidden_biases;
        //apply sigmoid
        for i in 0..test.len() {
            test[i] = sigmoid(test[i]);
        }
        test *= output_weights;
        test += output_biases;
        //apply sigmoid
        for i in 0..test.len() {
            test[i] = sigmoid(test[i]);
        }
    
        let len = test.len();
        for i in 0..len {
            // defmt::info!("{}: {}", i, test[i]);
            defmt::println!("{}: {}", i, test[i]);
        }

        defmt::println!("---------------------------------------------------------------");
        // defmt::info!("\n");
        // let res = test1_mat + test2_mat;
        // let random_value = random_f32();

        // defmt::info!("Random value: {}", random_value);

        // defmt::info!("input: {}", defmt::Display2Format(&input.len()));
        // defmt::info!("hidden_weights: {}", defmt::Display2Format(&hidden_weights.len()));
        // defmt::info!("hidden_biases: {}", defmt::Display2Format(&hidden_biases.len()));
        // defmt::info!("output_weights: {}", defmt::Display2Format(&output_weights.len()));
        // defmt::info!("output_biases: {}", defmt::Display2Format(&output_biases.len()));
        // defmt::info!("test: {}", defmt::Display2Format(&test.len()));
        // defmt::info!("Blink");



        Timer::after(Duration::from_millis(100)).await;
    }
}
