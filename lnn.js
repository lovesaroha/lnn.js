/*  Love Saroha
    lovesaroha1994@gmail.com (email address)
    https://www.lovesaroha.com (website)
    https://github.com/lovesaroha  (github)
*/
"use-strict";

(function ($) {

    // Tensor Object.
    class TensorObject {
        constructor() {
            // Create tensor object.
            this.shape = [];
            this.values = [];
        }

        // Print.
        Print() {
            // Check shape.
            switch (this.shape.length) {
                case 0:
                    console.log(`${this.values} \nScalar \n`);
                    return;
                case 2:
                    let printLog = ``;
                    for (let i = 0; i < this.shape[0]; i++) {
                        printLog = printLog + `[ `;
                        for (let j = 0; j < this.shape[1]; j++) {
                            printLog = printLog + ` ${this.values[i][j]} `;
                        }
                        printLog = printLog + ` ] \n`;
                    }
                    console.log(printLog + ` (${this.shape[0]}x${this.shape[1]}) \n`);
            }
        }

        // Copy tensor.
        Copy() {
            let newTensor = new TensorObject();
            newTensor.shape = JSON.parse(JSON.stringify(this.shape));
            newTensor.values = JSON.parse(JSON.stringify(this.values));
            return newTensor;
        }

        // Addition.
        Add(arg) {
            return elementWise(this, arg, 0);
        }
        // Subtraction.
        Sub(arg) {
            return elementWise(this, arg, 1);
        }
        // Multiplication.
        Mul(arg) {
            return elementWise(this, arg, 2);
        }
        // Square.
        Square() {
            return elementWise(this, this, 2);
        }
        // Divide.
        Divide(arg) {
            return elementWise(this, arg, 3);
        }
        // Dot product.
        Dot(arg) {
            if (this.shape.length == 0 || arg.shape.length == 0 || arg instanceof TensorObject == false) {
                console.error(`lnn.js: Cannot perform dot product, Check arguments!`);
                return this;
            }
            if (this.shape[1] != arg.shape[0]) {
                console.error("lnn.js: Number of columns of first must be equal to number of rows in second!");
                return this;
            }
            let newTensor = new TensorObject();
            newTensor.shape = [this.shape[0], arg.shape[1]];
            for (let i = 0; i < newTensor.shape[0]; i++) {
                let row = [];
                for (let j = 0; j < newTensor.shape[1]; j++) {
                    let sum = 0;
                    for (let r = 0; r < this.shape[1]; r++) {
                        sum += this.values[i][r] * arg.values[r][j];
                    }
                    row.push(sum);
                }
                newTensor.values.push(row);
            }
            return newTensor;
        }
        // Add all values.
        Sum() {
            let sum = 0;
            switch (this.shape.length) {
                case 0:
                    return this.values;
                    break;
                case 2:
                    for (let i = 0; i < this.shape[0]; i++) {
                        for (let j = 0; j < this.shape[1]; j++) {
                            sum += this.values[i][j];
                        }
                    }
            }
            return sum;
        }

        // Map function.
        Map(callback) {
            let newTensor = this.Copy();
            // Checks for valid callback function.
            if (callback && typeof callback != "function") {
                return newTensor;
            }
            // Check shape.
            switch (this.shape.length) {
                case 0:
                    newTensor.values = callback(newTensor.values);
                    break;
                case 2: // Matrix
                    for (let i = 0; i < this.shape[0]; i++) {
                        for (let j = 0; j < this.shape[1]; j++) {
                            newTensor.values[i][j] = callback(newTensor.values[i][j]);
                        }
                    }
            }
            return newTensor;
        }

        // Transpose. 
        Transpose() {
            let newTensor = this.Copy();
            switch (this.shape.length) {
                case 0:
                    return newTensor;
                case 2:
                    newTensor.shape = [this.shape[1], this.shape[0]];
                    newTensor.values = [];
                    for (let i = 0; i < newTensor.shape[0]; i++) {
                        let r = [];
                        for (let j = 0; j < newTensor.shape[1]; j++) {
                            r.push(this.values[j][i]);
                        }
                        newTensor.values.push(r);
                    }
            }
            return newTensor;
        }

        // Add column values.
        AddCols() {
            let newTensor = new TensorObject();
            // Check shape.
            switch (this.shape.length) {
                case 0:
                    return this.Copy();
                case 2:
                    newTensor.shape = [this.shape[0], 1];
                    newTensor.values = [];
                    for (let i = 0; i < this.shape[0]; i++) {
                        let sum = 0;
                        for (let j = 0; j < this.shape[1]; j++) {
                            sum += this.values[i][j];
                        }
                        newTensor.values.push([sum]);
                    }
            }
            return newTensor;
        }
        // Col extend.
        ColExtend(scale) {
            let newTensor = this.Copy();
            // Check shape.
            switch (this.shape.length) {
                case 0:
                    return this;
                case 2:
                    newTensor.shape = [newTensor.shape[0], newTensor.shape[1] * scale];
                    newTensor.values = [];
                    for (let i = 0; i < newTensor.shape[0]; i++) {
                        let r = [];
                        for (let j = 0; j < newTensor.shape[1]; j++) {
                            r.push(this.values[i][parseInt(j / scale)]);
                        }
                        newTensor.values.push(r);
                    }
            }
            return newTensor;
        }
        // Make batches.
        MakeBatches(size) {
            let newTensor = [];
            // Check shape.
            switch (this.shape.length) {
                case 0:
                    return [this.Copy()];
                case 2:
                    let totalBatches = this.shape[1] / size;
                    if (totalBatches * size != this.shape[1]) {
                        totalBatches += 1;
                    }
                    let initial = size;
                    for (let t = 0; t < totalBatches; t++) {
                        initial = t * size;
                        let limit = initial + size;
                        if (limit > this.shape[1]) {
                            limit = this.shape[1];
                        }
                        let nts = Tensor([this.shape[0], limit - initial]);
                        nts.values = [];
                        for (let i = 0; i < this.shape[0]; i++) {
                            let r = [];
                            for (let j = initial; j < limit; j++) {
                                r.push(this.values[i][j]);
                            }
                            nts.values.push(r);
                        }
                        newTensor.push(nts);
                    }
                    break;
                default:
                    return [this.Copy()];
            }
            return newTensor;
        }
    }

    // Tensor.
    function Tensor(shape, minimum, maximum) {
        let ts = new TensorObject();
        if (!shape || shape.constructor != Array) { return ts; }
        if (shape.length == 0) {
            ts.shape = [];
        } else if (shape.length == 1) {
            ts.shape = [shape[0], 1];
        } else {
            ts.shape = [shape[0], shape[1]];
        }
        // Create a tensor.
        switch (ts.shape.length) {
            case 0:
                ts.values = Random(minimum, maximum);
                break;
            case 2:
                for (let i = 0; i < ts.shape[0]; i++) {
                    let row = [];
                    for (let j = 0; j < ts.shape[1]; j++) {
                        row.push(Random(minimum, maximum));
                    }
                    ts.values.push(row);
                }
        }
        return ts;
    }

    // This function convert to tensor.
    function ToTensor(values) {
        let newTensor = new TensorObject();
        if (typeof values == "number") {
            // Scalar.
            newTensor.values = values;
        } else if (values.constructor == Array) {
            if (values[0].constructor == Array) {
                // Matrix.
                newTensor.shape = [values.length, values[0].length];
                for (let i = 0; i < newTensor.shape[0]; i++) {
                    let row = [];
                    for (let j = 0; j < newTensor.shape[1]; j++) {
                        if (typeof values[i][j] != "number") {
                            console.error(`lnn.js: Array contains invalid number at ${i}${j} in ToTensor()!`);
                            return newTensor;
                        }
                        row.push(values[i][j]);
                    }
                    newTensor.values.push(row);
                }
            } else {
                // Vector.
                newTensor.shape = [values.length, 1];
                for (let i = 0; i < newTensor.shape[0]; i++) {
                    if (typeof values[i] != "number") {
                        console.error(`lnn.js: Array contains invalid number at ${i} in ToTensor()!`);
                        return newTensor;
                    }
                    newTensor.values.push([values[i]]);
                }
            }
        }
        return newTensor;
    }

    // Random number between range.
    function Random(min, max) {
        // Checks if min or max are given.
        if (!min || typeof min != "number") {
            min = 0;
        }
        if (!max || typeof min != "number") {
            max = 0;
        }
        let randNumb = Math.random();
        if (min < 0) {
            randNumb *= (max + Math.abs(min));
        } else {
            randNumb *= (max - min);
        }
        randNumb += min;
        return randNumb;
    }

    // Sigmoid function.
    function Sigmoid(x) {
        // If x is not a number.
        if (typeof x != "number") {
            console.error("lnn.js: Input to sigmoid() is not a valid number!");
            return 0;
        }
        return 1 / (1 + Math.exp(-x));
    }
    // Differential of sigmoid.
    function Dsigmoid(y) {
        // If y is not a number.
        if (typeof y != "number") {
            console.error("lnn.js: Input to dsigmoid() is not a valid number!");
            return 0;
        }
        return y * (1 - y);
    }

    // Relu function.
    function Relu(x) {
        // If x is not a number.
        if (typeof x != "number") {
            console.error("lnn.js: Input to relu() is not a valid number!");
            return 0;
        }
        return Math.max(0, x);
    }

    // Differentiation of relu.
    function Drelu(y) {
        // If y is not a number.
        if (typeof y != "number") {
            console.error("lnn.js: Input to drelu() is not a valid number!");
            return 0;
        }
        if (y <= 0) {
            return 0;
        } else {
            return 1;
        }
    }

    // Mean square error.
    function meanSquareError(predict, output, batchSize) {
        let newTensor = new TensorObject();
        newTensor.shape = [];
        newTensor.values = predict.Sub(output).Square().AddAll() / batchSize.values;
        return newTensor;
    }

    // Element wise operation.
    function elementWise(ts, arg, operation) {
        if (ts.shape.length == 0 && arg.shape.length == 0) {
            return scalarOperation(ts, arg, operation);
        } else if (ts.shape.length == 0 && arg.shape.length == 2) {
            return elementWiseWithMatrix(arg, ts, operation);
        }
        return elementWiseWithMatrix(ts, arg, operation);
    }

    // Scalar operation.
    function scalarOperation(ts, arg, operation) {
        let newTensor = ts.Copy();
        switch (operation) {
            case 0:
                newTensor.values += arg.values;
                break;
            case 1:
                newTensor.values -= arg.values;
                break;
            case 2:
                newTensor.values *= arg.values;
                break;
            case 3:
                newTensor.values /= arg.values;
        }
        return newTensor;
    }
    // Element wise with matrix.
    function elementWiseWithMatrix(ts, arg, operation) {
        let newTensor = ts.Copy();
        switch (arg.shape.length) {
            case 0:
                for (let i = 0; i < newTensor.shape[0]; i++) {
                    for (let j = 0; j < newTensor.shape[1]; j++) {
                        switch (operation) {
                            case 0:
                                newTensor.values[i][j] += arg.values;
                                break;
                            case 1:
                                newTensor.values[i][j] -= arg.values;
                                break;
                            case 2:
                                newTensor.values[i][j] *= arg.values;
                                break;
                            case 3:
                                newTensor.values[i][j] /= arg.values;
                        }
                    }
                }
                break;
            case 2:
                // Check dimensions for element wise and extend columns.
                if (newTensor.shape[0] == arg.shape[0] && newTensor.shape[1] != arg.shape[1]) {
                    if (arg.shape[1] < newTensor.shape[1]) {
                        arg = arg.ColExtend(newTensor.shape[1]);
                    } else {
                        newTensor = newTensor.ColExtend(newTensor.shape[1]);
                        newTensor.shape = [arg.shape[0], arg.shape[1]];
                    }
                }
                for (let i = 0; i < newTensor.shape[0]; i++) {
                    for (let j = 0; j < newTensor.shape[1]; j++) {
                        switch (operation) {
                            case 0:
                                newTensor.values[i][j] = newTensor.values[i][j] + arg.values[i][j];
                                break;
                            case 1:
                                newTensor.values[i][j] = newTensor.values[i][j] - arg.values[i][j];
                                break;
                            case 2:
                                newTensor.values[i][j] = newTensor.values[i][j] * arg.values[i][j];
                                break;
                            case 3:
                                newTensor.values[i][j] = newTensor.values[i][j] / arg.values[i][j];
                        }
                    }
                }
        }
        return newTensor;
    }

    // Model object.
    class ModelObject {
        constructor() {
            this.layers = [];
            this.loss = {};
            this.optimizer = {};
        }
        // Add layer.
        AddLayer(config) {
            if (!config) { return; }
            let inputSize = 1;
            let layer = new LayerObject();
            if (this.layers.length == 0) {
                if (!config.inputShape) {
                    return;
                } else if (config.inputShape.length == 0) {
                    return;
                }
                for (let i = 0; i < config.inputShape.length; i++) {
                    inputSize *= config.inputShape[i];
                }
                if (!config.units || config.units == 0) {
                    config.units = inputSize;
                }
            } else {
                // Not first layer.
                if (!config.units || config.units == 0) {
                    config.units = this.layers[this.layers.length - 1].units;
                }
                inputSize = this.layers[this.layers.length - 1].units;
            }
            // Set activation function.
            switch (config.activation) {
                case "sigmoid":
                    layer.activationFunc = Sigmoid;
                    layer.dactivationFunc = Dsigmoid;
                    break;
                case "relu":
                    layer.activationFunc = Relu;
                    layer.dactivationFunc = Drelu;
                    break;
                default:
                    layer.activationFunc = Sigmoid;
                    layer.dactivationFunc = Dsigmoid;
            }
            layer.units = config.units;
            layer.inputSize = inputSize;
            // Add layer.
            this.layers.push(layer);
        }

        // Make model.
        Make(config) {
            if (!config) { return; }
            // Set loss function.
            switch (config.loss) {
                case "meanSquareError":
                    this.loss = meanSquareError;
                    break;
                default:
                    this.loss = meanSquareError;
            }
            // Set optimizer.
            switch (config.optimizer) {
                case "sgd":
                    this.optimizer = gradientDescent;
                    break;
                default:
                    this.optimizer = gradientDescent;
            }
            // Set learning rate.
            if (!config.learingRate || config.learingRate == 0 || config.learingRate > 1 || typeof config.learingRate != "number") {
                this.learingRate = ToTensor(0.2);
            } else {
                this.learingRate = ToTensor(config.learingRate);
            }
            // Create weights and biases in layers.
            for (let i = 0; i < this.layers.length; i++) {
                this.layers[i].weights = Tensor([this.layers[i].units, this.layers[i].inputSize], -1, 1);
                this.layers[i].biases = Tensor([this.layers[i].units, 1], -1, 1);
            }
        }
        // Predict.
        Predict(input) {
            if (!input || input instanceof TensorObject == false) { return; }
            let i = 0;
            // Forward propagation.
            for (i = 0; i < this.layers.length; i++) {
                this.layers[i].weightedSum = this.layers[i].weights.Dot(input).Add(this.layers[i].biases);
                this.layers[i].output = this.layers[i].weightedSum.Map(this.layers[i].activationFunc);
                input = this.layers[i].output;
            }
            return this.layers[i - 1].output;
        }
        // Train.
        Train(inputs, outputs, config) {
            // Check arguments.
            if (!inputs || inputs instanceof TensorObject == false || !outputs || outputs instanceof TensorObject == false) { return; }
            // Check config
            if (!config) {
                config = { batchSize: 1, epochs: 100 };
            }
            if (!config.batchSize || typeof config.batchSize != "number" || config.batchSize < 1) {
                config.batchSize = 1;
            }
            if (!config.epochs || typeof config.epochs != "number" || config.epochs < 1) {
                config.epochs = 100;
            }
            // Make batches.
            let inputBatches = inputs.MakeBatches(config.batchSize);
            let outputBatches = outputs.MakeBatches(config.batchSize);
            // For each epochs.
            for (let i = 0; i < config.epochs; i++) {
                // If shuffle.
                if (config.shuffle) {
                    shuffle(inputBatches, outputBatches);
                }
                // For each batch.
                for (let b = 0; b < inputBatches.length; b++) {
                    // Take a batch and predict.
                    let batchOutput = this.Predict(inputBatches[b]);
                    let batchSize = Tensor([], inputBatches[b].shape[1]);
                    // Run optimizer.
                    this.optimizer(this, batchOutput, batchSize, inputBatches[b], outputBatches[b]);
                }
            }
        }
    }

    // Layer object.
    class LayerObject {
        constructor() {
            this.weights = [];
            this.dweights = [];
            this.biases = [];
            this.dbiases = [];
            this.output = [];
            this.doutput = [];
            this.weightedSum = [];
            this.dweightedSum = [];
            this.units = 1;
            this.activationFunc = {};
            this.dactivationFunc = {};
            this.inputSize = 1;
        }
    }

    // Shuffle.
    function shuffle(arg1, arg2) {
        for (let i = 0; i < arg1.length; i++) {
            let p1 = Random(0, arg1.length, true);
            let p2 = Random(0, arg1.length, true);
            let temp1 = arg1[p1];
            let temp2 = arg2[p1];
            arg1[p1] = arg1[p2];
            arg2[p1] = arg2[p2];
            arg1[p2] = temp1;
            arg2[p2] = temp2;
        }
    }

    // Gradient descent.
    function gradientDescent(m, batchOutput, batchSize, inputArg, output) {
        let lIndex = m.layers.length - 1;
        let input = inputArg;
        if (m.layers.length > 1) {
            input = m.layers[lIndex - 1].output;
        }
        m.layers[lIndex].dweightedSum = batchOutput.Sub(output);
        m.layers[lIndex].dweights = m.layers[lIndex].dweightedSum.Dot(input.Transpose()).Divide(batchSize);
        m.layers[lIndex].dbiases = m.layers[lIndex].dweightedSum.AddCols().Divide(batchSize);
        m.layers[lIndex].weights = m.layers[lIndex].weights.Sub(m.layers[lIndex].dweights.Mul(m.learingRate));
        m.layers[lIndex].biases = m.layers[lIndex].biases.Sub(m.layers[lIndex].dbiases.Mul(m.learingRate));
        // Back propagation.
        for (let j = lIndex - 1; j >= 0; j--) {
            if (j == 0) {
                input = inputArg;
            } else {
                input = m.layers[j - 1].output;
            }
            m.layers[j].doutput = m.layers[j+1].weights.Transpose().Dot(m.layers[j+1].dweightedSum);
            m.layers[j].dweightedSum = m.layers[j].doutput.Mul(m.layers[j].output.Map(m.layers[j].dactivationFunc));
            m.layers[j].dbiases = m.layers[j].dweightedSum.AddCols().Divide(batchSize);
            m.layers[j].dweights = m.layers[j].dweightedSum.Dot(input.Transpose()).Divide(batchSize);
            m.layers[j].weights = m.layers[j].weights.Sub(m.layers[j].dweights.Mul(m.learingRate));
            m.layers[j].biases = m.layers[j].biases.Sub(m.layers[j].dbiases.Mul(m.learingRate));
        }
    }

    // Exported model function.
    function Model() {
        return new ModelObject();
    }

    // Export functions.
    $.lnn = {
        Tensor: Tensor,
        ToTensor: ToTensor,
        Random: Random,
        Model: Model,
        Sigmoid: Sigmoid,
        Dsigmoid: Dsigmoid,
        Relu: Relu,
        Drelu: Drelu,
        Shuffle: shuffle
    }
}(window));