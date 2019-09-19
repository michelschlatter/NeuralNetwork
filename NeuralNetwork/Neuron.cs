using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Neuron
    {
        public Neuron(Layer layer)
        {
            Layer = layer;
        }

        public Layer Layer { get; set; }
        public List<Connection> Connections { get; set; } = new List<Connection>();
        public double Net { get; set; }
        public bool IsBias { get; set; }
        public double Error { get; set; }
        public double Delta { get; set; }
        public double Out
        {
            get
            {
                if (IsBias)
                {
                    return 1;
                }
                if (Layer.Activation != null)
                {
                    return Layer.Activation.Calculate(Net);
                }
                return Net;
            }
        }


    }
}
