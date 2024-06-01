# nn-pico

This project is a handwritten digit recognition system implemented entirely on a Raspberry Pi Pico. It uses a simple neural network (NN) with one hidden layer consisting of 10 neurons. The NN is trained on the MNIST dataset, a large database of handwritten digits that is commonly used for training various image processing systems.

# Image
![image](https://github.com/DuckyBlender/nn-pico/assets/42645784/7bb2d7c1-5349-47b8-a2af-b235a385dbab)

The example input is a handwritten 0. The NN clearly chooses 0.

## Features

- The system is capable of recognizing handwritten digits.
- The neural network is trained on the MNIST dataset.
- The entire system runs on a Raspberry Pi Pico, a microcontroller board with 264KB of RAM.
- The neural network uses only about 32KB of memory, making it highly efficient for such a constrained environment.
- The project uses the [Embassy](https://embassy.dev/) framework for Rust, which provides async/await support and other features for embedded systems programming.

## How it Works

The weights for the neural network are pre-computed using a Python script located in the `python/` directory. These weights are then used in the Rust program running on the Raspberry Pi Pico to recognize the digits.

The Rust program uses the nalgebra library for linear algebra operations, and the defmt library for efficient logging and debugging on embedded systems.

## Running the Project

To run the project, you will need to have Rust and the necessary toolchains installed. You can then clone the repository, build the project, and flash it onto your Raspberry Pi Pico.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
