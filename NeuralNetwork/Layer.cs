using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Layer
    {
        public Layer(int neurons, IActivation activation = null, bool addBias = false)
        {
            Activation = activation;

            for (int i = 0; i < neurons; i++)
            {
                Neurons.Add(new Neuron(this));
            }
            if (addBias)
            {
                Neurons.Add(new Neuron(this) { IsBias = true });
            }
        }

        public IActivation Activation { get; set; }
        public List<Neuron> Neurons { get; set; } = new List<Neuron>();
        public string Metainformation { get; set; }
        public Layer LayerBefore { get; set; }
        public Layer LayerAfter { get; set; }

        public void FeedForward(double[] input)
        {

            foreach (Neuron n in Neurons)
            {
                n.Net = input[Neurons.IndexOf(n)];
            }

            if (LayerAfter != null)
            {
                double[] newInput = new double[LayerAfter.Neurons.Count];

                foreach (Neuron n in LayerAfter.Neurons)
                {
                    double netSum = 0;
                    foreach (Connection c in n.Connections)
                    {
                        double neuronValue = c.FromNeuron.Out;
                        netSum += c.Weight * neuronValue;
                    }
                    newInput[LayerAfter.Neurons.IndexOf(n)] = netSum;
                }

                LayerAfter.FeedForward(newInput);
            }
        }
        
    }
}
