

function buildModel () {

  const model = tf.sequential({
    layers: [
      tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
      tf.layers.dense({units: 10, activation: 'softmax'}),
    ]
  });

  const elem = document.createElement('p');

  model.weights.forEach(w => {
    const txt = document.createTextNode(`${w.name} ${w.shape}\n`);
    elem.appendChild(txt);
    document.getElementById('deep-image-prior-container').appendChild(elem);
  });



  model.weights.forEach(w => {
    const newVals = tf.randomNormal(w.shape);
    // w.val is an instance of tf.Variable
    w.val.assign(newVals);
  });


  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;

}

function trainModel(model) {

  const data = tf.randomNormal([100, 784]);
  const labels = tf.randomUniform([100, 10]);

  function onBatchEnd(batch, logs) {
    console.log('Accuracy', logs.acc);
  }

  model.fit(data, labels, {
    epochs: 5,
    batchSize: 32,
    callbacks: {onBatchEnd}
  }).then(info => {
    console.log('Final accuracy', info.history.acc);
  });

  return model;

}

