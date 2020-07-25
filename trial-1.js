/* 
    Simple and straightforward
    EVERYTHING IN ONE FILE FOR NOW
    REDUCE COMMENTS

    Load files - convert to tensor
    input to model.fit tensor4D and tensor2D
    model fit
 */

const tf = require('@tensorflow/tfjs-node');

const fs = require('fs');
const path = require('path');

const train_dir = path.resolve('./train_dir');

// # Directory with our training cat/dog pictures
const train_cats_dir = path.join(train_dir, 'cats')
const train_dogs_dir = path.join(train_dir, 'dogs')


const train_cats_filenames  = [
    'cat.0.jpg',   'cat.1.jpg',   'cat.10.jpg',  'cat.2.jpg',   'cat.259.jpg',
    'cat.260.jpg', 'cat.261.jpg', 'cat.262.jpg', 'cat.263.jpg', 'cat.264.jpg',
    'cat.265.jpg', 'cat.266.jpg', 'cat.267.jpg', 'cat.268.jpg', 'cat.269.jpg',
    'cat.270.jpg', 'cat.271.jpg', 'cat.272.jpg', 'cat.273.jpg', 'cat.274.jpg',
    'cat.275.jpg', 'cat.276.jpg', 'cat.277.jpg', 'cat.278.jpg', 'cat.279.jpg',
    'cat.280.jpg', 'cat.281.jpg', 'cat.282.jpg', 'cat.283.jpg', 'cat.284.jpg',
    'cat.285.jpg', 'cat.286.jpg', 'cat.287.jpg', 'cat.288.jpg', 'cat.3.jpg',
    'cat.4.jpg',   'cat.495.jpg', 'cat.496.jpg', 'cat.497.jpg', 'cat.498.jpg',
    'cat.499.jpg', 'cat.5.jpg',   'cat.500.jpg', 'cat.501.jpg', 'cat.502.jpg',
    'cat.503.jpg', 'cat.504.jpg', 'cat.505.jpg', 'cat.506.jpg', 'cat.507.jpg',
    'cat.508.jpg', 'cat.509.jpg', 'cat.510.jpg', 'cat.511.jpg', 'cat.512.jpg',
    'cat.513.jpg', 'cat.57.jpg',  'cat.58.jpg',  'cat.59.jpg',  'cat.6.jpg',
    'cat.60.jpg',  'cat.61.jpg',  'cat.62.jpg',  'cat.63.jpg',  'cat.64.jpg',
    'cat.65.jpg',  'cat.66.jpg',  'cat.67.jpg',  'cat.68.jpg',  'cat.69.jpg',
    'cat.7.jpg',   'cat.70.jpg',  'cat.71.jpg',  'cat.72.jpg',  'cat.73.jpg',
    'cat.74.jpg',  'cat.75.jpg',  'cat.76.jpg',  'cat.8.jpg',   'cat.9.jpg',
    'cat.925.jpg', 'cat.926.jpg', 'cat.927.jpg', 'cat.928.jpg', 'cat.929.jpg',
    'cat.930.jpg', 'cat.931.jpg', 'cat.932.jpg', 'cat.933.jpg', 'cat.934.jpg',
    'cat.935.jpg', 'cat.936.jpg', 'cat.937.jpg', 'cat.938.jpg', 'cat.939.jpg',
    'cat.940.jpg', 'cat.941.jpg', 'cat.942.jpg', 'cat.943.jpg', 'cat.944.jpg'
  ]
  
const train_dogs_filenames  = [
    'dog.256.jpg', 'dog.257.jpg', 'dog.258.jpg', 'dog.259.jpg', 'dog.260.jpg',
    'dog.261.jpg', 'dog.262.jpg', 'dog.263.jpg', 'dog.264.jpg', 'dog.265.jpg',
    'dog.266.jpg', 'dog.267.jpg', 'dog.268.jpg', 'dog.269.jpg', 'dog.270.jpg',
    'dog.271.jpg', 'dog.272.jpg', 'dog.273.jpg', 'dog.274.jpg', 'dog.275.jpg',
    'dog.276.jpg', 'dog.277.jpg', 'dog.278.jpg', 'dog.279.jpg', 'dog.280.jpg',
    'dog.281.jpg', 'dog.282.jpg', 'dog.283.jpg', 'dog.284.jpg', 'dog.285.jpg',
    'dog.468.jpg', 'dog.469.jpg', 'dog.47.jpg',  'dog.470.jpg', 'dog.471.jpg',
    'dog.472.jpg', 'dog.473.jpg', 'dog.474.jpg', 'dog.475.jpg', 'dog.476.jpg',
    'dog.477.jpg', 'dog.478.jpg', 'dog.479.jpg', 'dog.48.jpg',  'dog.480.jpg',
    'dog.481.jpg', 'dog.482.jpg', 'dog.483.jpg', 'dog.484.jpg', 'dog.485.jpg',
    'dog.486.jpg', 'dog.487.jpg', 'dog.488.jpg', 'dog.489.jpg', 'dog.49.jpg',
    'dog.490.jpg', 'dog.491.jpg', 'dog.492.jpg', 'dog.493.jpg', 'dog.494.jpg',
    'dog.495.jpg', 'dog.496.jpg', 'dog.497.jpg', 'dog.50.jpg',  'dog.51.jpg',
    'dog.52.jpg',  'dog.53.jpg',  'dog.54.jpg',  'dog.55.jpg',  'dog.56.jpg',
    'dog.934.jpg', 'dog.935.jpg', 'dog.936.jpg', 'dog.937.jpg', 'dog.938.jpg',
    'dog.939.jpg', 'dog.940.jpg', 'dog.941.jpg', 'dog.942.jpg', 'dog.943.jpg',
    'dog.944.jpg', 'dog.945.jpg', 'dog.946.jpg', 'dog.947.jpg', 'dog.948.jpg',
    'dog.949.jpg', 'dog.950.jpg', 'dog.951.jpg', 'dog.952.jpg', 'dog.953.jpg',
    'dog.954.jpg', 'dog.955.jpg', 'dog.956.jpg', 'dog.957.jpg', 'dog.958.jpg',
    'dog.959.jpg', 'dog.960.jpg', 'dog.961.jpg', 'dog.962.jpg', 'dog.963.jpg'
  ]
// MAX 100
numFilesToUse = 20

xs = []
ys = []

for (let i=0; i<numFilesToUse; i++) {
    const filePath = path.join(train_cats_dir, train_cats_filenames[i])
    const x = tf.tidy( () => {
        
        let x = fs.readFileSync(filePath)
    
        // x = tf.node.decodeJpeg(UiAnt8Array.from(x))
        x = tf.node.decodeJpeg(x)

        x = tf.image.resizeNearestNeighbor(x, [150, 150])

        x = x.div(tf.scalar(255.0))
        
        return x
    })
    xs.push(x)

    ys.push([0])

}


for (let i=0; i<numFilesToUse; i++) {
    const filePath = path.join(train_dogs_dir, train_dogs_filenames[i])
    const x = tf.tidy( () => {
        
        let x = fs.readFileSync(filePath)
    
        // x = tf.node.decodeJpeg(Uint8Array.from(x))
        x = tf.node.decodeJpeg(x)

        x = tf.image.resizeNearestNeighbor(x, [150, 150])

        // console.log(x)
        x = x.div(tf.scalar(255.0))
        
        return x
    })

    xs.push(x)

    ys.push([1])
}

xs = tf.stack(xs)
ys = tf.tensor2d(ys, [numFilesToUse*2, 1], 'int32')


const model = tf.sequential({
    layers: [
        tf.layers.conv2d({filters: 16, kernelSize: [3, 3], activation:'relu', inputShape:[150, 150, 3]}),
        tf.layers.maxPooling2d({poolSize: [2, 2]}),
        
        tf.layers.conv2d({filters: 32, kernelSize: [3,3], activation: 'relu', }),
        tf.layers.maxPooling2d({poolSize: [2, 2]}),
    
        tf.layers.conv2d({filters: 64, kernelSize: [3,3], activation: 'relu', }),
        tf.layers.maxPooling2d({poolSize: [2, 2]}),
    
        tf.layers.flatten(),
        tf.layers.dense({units: 512, activation: 'relu'}),
        tf.layers.dense({units: 1, activation: 'sigmoid'})
    ],

});


model.compile({
    optimizer: tf.train.rmsprop(0.001),
    loss: 'binaryCrossentropy', //'categoricalCrossentropy', //'binaryCrossentropy', //'sigmoidCrossEntropy',
    metrics: ['accuracy']
});

model.fit(
    xs, ys, {
        epochs: 20,
    }
)
.catch(err => {
    console.log(err)
})