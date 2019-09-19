using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Connection
    {
        public Neuron FromNeuron { get; set; }
        public Neuron ToNeuron { get; set; }
        public double WeightUpdateBefore { get; set; }
        public double Weight { get; set; }
        public string MetaInformation { get; set; }
    }
}
