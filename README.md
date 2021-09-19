# lnn.js
This is a generalized neural network package with clean and transparent API for the javascript. Also available for golang [github/lovesaroha/lnn](https://github.com/lovesaroha/lnn) 

## Features
- Lightweight and Fast.
- Tensor Operations.
- Sequential Models.
- Support loss functions like (Mean Square Error).
- Opitmization algorithms like (Gradient Descent).

## Installation

```html
    <script type="text/javascript" src="lnn.js"></script>
```

## Tensor Usage

### Create Tensor

```js
  // Create tensor of given shape.
  let tensor = lnn.Tensor([3, 4]);
  // Print values.
  tensor.Print();
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/53.png)

### Random Tensor

```js
  // Create tensor of given shape and (minimum, maximum).
  let tensor = lnn.Tensor([3, 4] , -1 , 1);
  // Print values.
  tensor.Print();
  // Scalar tensor.
  let stensor = lnn.Tensor([], -1 , 1);
  // Print values.
  stensor.Print();
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/54.png)

### Convert Array Or Values Into Tensor

```js
    // Array to tensor and print values.
    lnn.ToTensor([1, 2, 3]).Print();
    // 2d array to tensor and print values.
    lnn.ToTensor([[1, 2],[3, 4]]).Print();
    // Value to tensor and print values.
    lnn.ToTensor(5).Print();
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/55.png)

### Tensor Element Wise Operations (Add, Subtract, Multiply, Divide) 

```js
  // Create a random tensor.
  let tensor = lnn.Tensor([3, 4] , 10 , 20);
  let tensorB = lnn.Tensor([3, 4] , 0 , 10);

  // Add and print values.
  tensor.Add(tensorB).Print();
  // Subtract and print values.
  tensor.Sub(tensorB).Print();
  // Multiply and print values.
  tensor.Mul(tensorB).Print();
  // Divide and print values.
  tensor.Sub(tensorB).Print();
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/56.png)

### Tensor Element Wise Operations With Scalar Value (Add, Subtract, Multiply, Divide) 

```js
  // Create a random tensor.
  let tensor = lnn.Tensor([3, 4] , 10 , 20);
  let tensorB = lnn.ToTensor(4);

  // Add and print values.
  tensor.Add(tensorB).Print();
  // Subtract and print values.
  tensor.Sub(tensorB).Print();
  // Multiply and print values.
  tensor.Mul(tensorB).Print();
  // Divide and print values.
  tensor.Sub(tensorB).Print();
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/57.png)

### Tensor Dot Product 

```js
  // Create a random tensor.
  let tensor = lnn.Tensor([3, 4] , 10 , 20);
  let tensorB = lnn.Tensor([4, 3] , 0 , 10);

  // Dot product and print values.
  tensor.Dot(tensorB).Print();
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/58.png)

### Tensor Transpose
```js
  // Create a random tensor.
  let tensor = lnn.Tensor([3, 1] , 10 , 20);

  // Print values.
  tensor.Print();
  // Transpose and print values.
  tensor.Transpose().Print();
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/59.png)

### Add All Column Values
```js
  // Create a random tensor.
  let tensor = lnn.Tensor([3, 3] , 10 , 20);

  // Print values.
  tensor.Print();
  // Add columns and print values.
  tensor.AddCols().Print();
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/60.png)

### Change Tensor Values (Map)
```js
  // Create a random tensor.
  let tensor = lnn.Tensor([3, 3] , 0, 10);

  // Print values.
  tensor.Print();
  // Square and print values.
  tensor.Map(function(value) {
    return value * value
  }).Print();
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/61.png)

## Model Usage

### Create Model
```js
  // Create a model.
  let model = lnn.Model();
```

### Add Layers In Model
```js
  // Add layer to model with 4 units , Input shape (2) and activation function sigmoid.
  model.AddLayer({inputShape: [2], units: 4, activation: "sigmoid"});
  // Add another layer to model with 1 unit and activation function sigmoid.
  model.AddLayer({units: 1});
```

### Model Configuration
```js
  // Makes the model with given values of loss function , optimizer and learning rate.
  model.Make({loss: "meanSquareError", optimizer: "sgd", learningRate: 0.2});
```

### Train Model
```js
  // Trains the model with given configuration.
  model.Train(inputs, outputs, {epochs: 1000, batchSize: 4, shuffle: true});
```

### Predict Output
```js
  model.Predict(inputs);
```
## Examples

### XOR Gate Model Training

```js
  // Create a model.
  let model = lnn.Model()
  
  // Add layer to model with 4 units , Input shape (2) and activation function sigmoid.
  model.AddLayer({inputShape: [2], units: 4, activation: "sigmoid"});
  
  // Add another layer to model with 1 unit and activation function sigmoid.
  model.AddLayer({units: 1});
  
  // Makes the model with given values of loss function , optimizer and learning rate.
  model.Make({loss: "meanSquareError", optimizer: "sgd", learningRate: 0.2});
  
  // Inputs and outputs as a tensor object.
  let inputs = lnn.ToTensor([[1, 1, 0, 0],[1, 0, 1, 0]]);
  let outputs = lnn.ToTensor([[0, 1, 1, 0]]);
  
  // Trains the model with given configuration.
    model.Train(inputs, outputs, {epochs: 5000, batchSize: 4, shuffle: true});
  
  // Print values.
  model.Predict(inputs).Print();
```

![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/62.png)

### Logistic Regression (OR Gate)  With Tensor
```js 
  // Learning rate.
  let learningRate = lnn.ToTensor(0.2);
  let size = lnn.ToTensor(4);
  
  // Inputs and outputs as a tensor object.
  let inputs = lnn.ToTensor([[1, 1, 0, 0],[1, 0, 1, 0]]);
  let outputs = lnn.ToTensor([[1, 1, 1, 0]]);

  // Weights and bias.
  let weights = lnn.Tensor([2, 1] , -1, 1);
  let bias = lnn.Tensor([], -1, 1);

  // Train weights and bias (epochs 1000).
  for(let i = 0; i < 1000; i++) {
    // Sigmoid(wx + b).
    let prediction = weights.Transpose().Dot(inputs).Add(bias).Map(lnn.Sigmoid);
    let dZ = prediction.Sub(outputs);
    weights = weights.Sub(inputs.Dot(dZ.Transpose()).Divide(size).Mul(learningRate));
    bias = bias.Sub(dZ.AddCols().Divide(size).Mul(learningRate));
  }

  // Show prediction.
  weights.Transpose().Dot(inputs).Add(bias).Map(lnn.Sigmoid).Print();

```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/63.png)