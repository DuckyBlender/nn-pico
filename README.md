# nn-pico

This project is a simple, yet powerful, Neural Network (NN) written in Rust, designed to recognize hand-written numbers. It's trained on the MNIST dataset and runs entirely on a Raspberry Pi Pico using the [Embassy](https://embassy.dev/) framework.

This project was written in a single day by me and [@F1L1Pv2](https://github.com/f1l1pv2)

## Image
![image](https://github.com/DuckyBlender/nn-pico/assets/42645784/7bb2d7c1-5349-47b8-a2af-b235a385dbab)

The example input is a handwritten 0. The NN clearly chooses 0.

## Features

- **Small Footprint**: The NN uses only about 32KB of the available 264KB RAM on the Raspberry Pi Pico.
- **Efficient**: Despite its small size, the NN boasts an impressive accuracy of about 92%.
- **Fast**: The performance is practically instant, making it suitable for real-time applications.
- **Custom Implementation**: The Rust program uses a custom sigmoid implementation and softmax for the output.

## How it Works

The weights for the NN are pre-computed using a Python script located in the `python/` directory. This script uses the Keras library for training the NN and generates the `weights.rs` file used by the Rust program.

The NN itself consists of one hidden layer with 10 neurons. Despite its simplicity, it delivers impressive results.

## Performance

The performance of this NN is practically instant. More detailed benchmarks will be coming soon.

## Future Work

We're always looking to improve and expand on this project. If you have any suggestions or want to contribute, feel free to open an issue or submit a pull request.

## License

This project is open source under the MIT license.