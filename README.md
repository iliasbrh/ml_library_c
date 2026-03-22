First C project : handmade ML library to train on MNIST, using only stdlib and math.h .

Reached 90.53% accuracy in 37s, which could be enhanced by implementing Adam optimizer, by forwarding images through batches, by adding dropout, and by using a learning rate scheduler.
Nothing here is optimized for performance, it is rather a simple implementation of an autograd and gradient descent.

Might want to improve the library in the future to have convolutional layers to make a GAN.


To compile, use
- gcc arena.c value.c matrix.c read_ubyte.c main.c -o model.exe

or whichever compiler you use.
