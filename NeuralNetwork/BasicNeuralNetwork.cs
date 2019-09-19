using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    public class BasicNeuralNetwork
    {
        Random random = new Random();

        private List<Layer> _layers = new List<Layer>();

        public List<Layer> Layers
        {
            get
            {
                return _layers;
            }
        }

        public void Serialize(string savePath)
        {
           string json =  JsonConvert.SerializeObject(this, new JsonSerializerSettings()
           {
               ReferenceLoopHandling = ReferenceLoopHandling.Ignore
           });
           File.WriteAllText(savePath, json);
        }

        public void AddLayer(Layer layer)
        {
            _layers.Add(layer);
            layer.Metainformation = $"Layer:{_layers.Count}";
            if (_layers.Count > 1)
            {
                int layerBeforeIdx = _layers.IndexOf(layer) - 1;
                Layer layerBefore = _layers[layerBeforeIdx];
                layerBefore.LayerAfter = layer;
                layer.LayerBefore = layerBefore;

                foreach (Neuron nlb in layerBefore.Neurons)
                {
                    foreach (Neuron n in layer.Neurons)
                    {
                        if (!n.IsBias)
                        {
                            Connection c = new Connection();
                            c.FromNeuron = nlb;
                            c.ToNeuron = n;
                            c.Weight = random.NextDouble() * (random.Next(0,2) >= 1 ? -1 : 1); // initialize weights between -4 and 4
                            c.MetaInformation = layer.Metainformation;
                            n.Connections.Add(c);
                        }
                    }
                }
            }
        }

        public List<double> Calculate(double[] input)
        {
            Layers[0].FeedForward(input);
            return Layers.Last().Neurons.Select(x => x.Out).ToList();
        }

        public void Train(List<double[]> inputs, List<double[]> labels, double learningRate, double minError, Action<int, double> callback)
        {
            double totalError = double.MaxValue;
            int iteration = 1;
            double mse = double.MaxValue;
            while (mse > minError)
            {
                totalError = 0;
                mse = 0;
                foreach (double[] input in inputs)
                {
                    Layers[0].FeedForward(input);

                    foreach (Neuron n in Layers.Last().Neurons)
                    {
                        double label = labels[inputs.IndexOf(input)][Layers.Last().Neurons.IndexOf(n)];
                        double output = n.Out;
                        
                        mse += Math.Pow(label - output, 2.0);
                        totalError += Math.Abs(label - output);
                    }
                    Backpropagate(labels[inputs.IndexOf(input)]);
                    UpdateWeights(learningRate);
                }
                mse = mse / inputs.Count;

                if (callback != null)
                {
                    callback(iteration, mse);
                }
                iteration++;
            }
        }

        private List<double> GetWeights()
        { 
            // maybe the index doesn't match witch the index in de js-algorithm - so check it!!!
            List<double> weights = new List<double>();
            foreach (Layer layer in Layers)
            {
                foreach(Neuron n in layer.Neurons)
                {
                    foreach(Connection c in n.Connections)
                    {
                        weights.Add(c.Weight);
                    }
                }
            }
            return weights;
        }

        // with help of this explanation of backprop http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
        private void Backpropagate(double[] labels)
        {
            for (int i = Layers.Count- 1; i >= 0; i--)
            {
                Layer layer = Layers[i];
                if (layer == Layers.Last()) // output layer
                {
                    foreach (Neuron n in layer.Neurons)
                    {
                        n.Delta = labels[layer.Neurons.IndexOf(n)] - n.Out; 
                    }
                }
                else if (layer != Layers.First()) // // hidden layer
                {
                    Layer aftereLayer = Layers[i + 1];
                    foreach (Neuron n in layer.Neurons)
                    {
                        double error = 0.0;

                        List<Connection> currentConnections = aftereLayer.Neurons.SelectMany(x => x.Connections).Where(x => x.FromNeuron == n).ToList();
                        foreach (Connection currentConnection in currentConnections)
                        {
                            error += currentConnection.ToNeuron.Delta * currentConnection.Weight;
                        }

                        n.Delta = error;
                    }
                }

            }

        }

        private void UpdateWeights(double lr, double momentum = 0.9)
        {
            foreach (Layer layer in Layers.OrderByDescending(x => x.Metainformation))
            {
                foreach (Neuron n in layer.Neurons)
                {
                    foreach (Connection c in n.Connections)
                    {
                        if (c.FromNeuron.IsBias)
                        {
                            double weightUpdate = (lr * c.ToNeuron.Delta * c.ToNeuron.Layer.Activation.Derivative(c.ToNeuron.Net)) + (c.WeightUpdateBefore * momentum);
                            c.Weight = c.Weight + weightUpdate;
                            c.WeightUpdateBefore = weightUpdate;
                        }
                        else
                        {
                            double weightUpdate = (lr * c.ToNeuron.Delta * c.ToNeuron.Layer.Activation.Derivative(c.ToNeuron.Net) * c.FromNeuron.Out) + (c.WeightUpdateBefore * momentum);
                            c.Weight = c.Weight + weightUpdate;
                            c.WeightUpdateBefore = weightUpdate;
                        }

                    }
                }

            }
        }

        #region Static
        public static BasicNeuralNetwork Load(string path)
        {
            BasicNeuralNetwork network = JsonConvert.DeserializeObject<BasicNeuralNetwork>(File.ReadAllText(path),
                new JsonSerializerSettings() { ReferenceLoopHandling = ReferenceLoopHandling.Ignore });
            return network;
        }
        #endregion
        
    }
}
