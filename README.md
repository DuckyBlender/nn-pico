# nn-pico

This project is a simple, yet powerful, Neural Network (NN) written in Rust, designed to recognize hand-written numbers. It's trained on the MNIST dataset and runs entirely on a Raspberry Pi Pico using the [Embassy](https://embassy.dev/) framework.

This project was written in a single day by me and [@F1L1Pv2](https://github.com/f1l1pv2)

## Image
![Code_WusMUlsp3N](https://github.com/DuckyBlender/nn-pico/assets/42645784/fab60485-4005-41b9-9ee7-b1ef868905ab)

We fed the NN a handwritten "0" and it correctly identified it!

## Features

- **Small Footprint**: The NN uses only about 32KB of the available 264KB RAM on the Raspberry Pi Pico.
- **Efficient**: Despite its small size, the NN boasts an impressive accuracy of about 92%.
- **Fast**: The performance is practically instant, making it suitable for real-time applications.
- **Custom Implementation**: The Rust program uses a custom sigmoid implementation and softmax for the output.

## How it Works

The weights for the NN are pre-computed using a Python script located in the `python/` directory. This script uses the Keras library for training the NN and generates the `weights.rs` file used by the Rust program.

The NN itself consists of one hidden layer with 10 neurons. Despite its simplicity, it delivers impressive results.

## Performance

The performance of this NN is impressive:
- Loading weights: 15ms
- Running the NN: 17ms
- Frame Rate: This NN could be ran at 30fps, making it suitable for real-time applications. Since it's single-core, there's potential for even more performance or accuracy improvements.

## Future Work

We're always looking to improve and expand on this project. Some ideas we'd like to explore:

- Create a web-based user interface to visualize and interact with the NN
- Wire the Pico to a touch screen to create a standalone, interactive demo
- Port this to a Garmin touch-screen watch

## License

This project is open source under the MIT license.
