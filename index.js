let recognizer;
let last100Vals = [];
let model;
let numClasses = 3;

async function app() {
  recognizer = speechCommands.create("BROWSER_FFT");
  await recognizer.ensureModelLoaded();
  //   predictWord();
  buildModel();
}

app();

// One frame is ~23ms of audio.
const NUM_FRAMES = 3;
// const NUM_FRAMES = 43;
let examples = []; //this will be an array of spectrograms

//three frames long
//corresponds to ~ 70ms samples
//232 is the num of individual discrete frequency needed to capture the human voice
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];

function buildModel() {
  model = tf.sequential(); //groups a linear stack of layers into a model
  // model.add(
  //   tf.layers.depthwiseConv2d({
  //     depthMultiplier: 8,
  //     kernelSize: [NUM_FRAMES, 3],
  //     activation: "relu",
  //     inputShape: INPUT_SHAPE
  //   })
  // );
  model.add(
    tf.layers.conv2d({
      inputShape: INPUT_SHAPE,
      filters: 32,
      kernelSize: [NUM_FRAMES, 3],
      activation: "relu",
      strides: (1, 1),
      padding: "same"
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }));
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: [NUM_FRAMES, 3],
      activation: "relu",
      strides: (1, 1),
      padding: "same" //"valid"
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: numClasses, activation: "tanh" }));
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });
}

function collect(label, toggle) {
  const temp = toggle[0];
  toggle[0] = { ...temp, label, on: !temp.on };

  if (recognizer.isListening()) {
    return recognizer.stopListening();
  }
  if (label == null) {
    return;
  }
  recognizer.listen(
    async ({ spectrogram: { frameSize, data } }) => {
      let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      examples.push({ vals, label });
      // console.log(examples[0]);
      document.querySelector(
        "#console"
      ).textContent = `${examples.length} examples collected`;
    },
    {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    }
  );
}

function normalize(x) {
  const mean = -100;
  const std = 10;
  return x.map(x => (x - mean) / std);
}

async function train() {
  toggleButtons(false);
  const ys = tf.oneHot(
    examples.map(e => e.label),
    numClasses
  );
  const xsShape = [examples.length, ...INPUT_SHAPE];
  const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.querySelector("#console").textContent = `Accuracy: ${(
          logs.acc * 100
        ).toFixed(1)}% Epoch: ${epoch + 1}`;
      }
    }
  });
  tf.dispose([xs, ys]);
  toggleButtons(true);
}

function flatten(tensors) {
  const size = tensors[0].length;
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));
  return result;
}

function toggleButtons(enable) {
  document.querySelectorAll("button").forEach(b => (b.disabled = !enable));
}

async function moveSlider(labelTensor, counter) {
  const label = (await labelTensor.data())[0];
  document.getElementById("console").textContent = label;

  let delta = 0.1;
  last100Vals.push(label);
  const prevValue = +document.getElementById("output").value;
  function mode(arr) {
    const counts = {};
    for (let i = 0; i < arr.length; i++) {
      let k = arr[i];
      counts[k] ? (counts[k] = counts[k] + 1) : (counts[k] = 1);
    }
    let mode = { highestVal: 0, mode: null };
    const keys = Object.keys(counts);
    keys.forEach(k =>
      counts[k] > mode.highestVal
        ? (mode = { highestVal: counts[k], mode: k })
        : null
    );
    return mode;
  }

  if (counter % 50 === 0) {
    const guess = mode(last100Vals);
    console.log(guess);
    document.getElementById("lastn").innerHTML = `<p>${guess.mode}</p>`;
    if (guess.mode == 0) document.body.style.backgroundColor = "red";
    if (guess.mode == 1) document.body.style.backgroundColor = "green";

    if (guess.mode == 2) {
      document.body.style.backgroundColor = "white";
      document.getElementById("lastn").innerHTML = "Waiting patiently...";
    }
    last100Vals = [];
  }
  counter % 50 == 0 ? console.log("THE LABEL", label) : null;

  document.getElementById("output").value =
    prevValue + (label === 0 ? -delta : delta);
}

async function listen() {
  if (recognizer.isListening()) {
    recognizer.stopListening();
    toggleButtons(true);
    document.getElementById("listen").textContent = "Listen";
    return;
  }
  toggleButtons(false);
  document.getElementById("listen").textContent = "Stop";
  document.getElementById("listen").disabled = false;
  var counter = 0;

  recognizer.listen(
    async ({ spectrogram: { frameSize, data } }) => {
      const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
      const probs = model.predict(input);
      const predLabel = probs.argMax(1);
      counter++;
      await moveSlider(predLabel, counter);
      tf.dispose([input, probs, predLabel]);
    },
    {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    }
  );
}

async function saveModel() {
  // await model.save("downloads://test-model1");
  const filename = document.getElementById("saveInput").value;
  await model
    .save(`localstorage://${filename}`)
    .then(r => console.log("done", r, "done"))
    .catch(console.log);
}

async function loadModel() {
  const log = arg => {
    console.log("done");
    return arg;
  };
  const filename = document.getElementById("loadInput").value;
  model = await tf.loadLayersModel(`localstorage://${filename}`).then(log);
}

//used in tutorial but we don't use for our example
// function predictWord() {
//   // Array of words that the recognizer is trained to recognize.
//   const words = recognizer.wordLabels();
//   recognizer.listen(
//     ({ scores }) => {
//       // Turn scores into a list of (score,word) pairs.
//       scores = Array.from(scores).map((s, i) => ({ score: s, word: words[i] }));
//       // Find the most probable word.
//       scores.sort((s1, s2) => s2.score - s1.score);
//       document.querySelector("#console").textContent = scores[0].word;
//     },
//     { probabilityThreshold: 0.75 }
//   );
// }
