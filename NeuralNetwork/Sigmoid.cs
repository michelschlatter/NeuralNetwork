using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Sigmoid : IActivation
    {
        public double Calculate(double value)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-value));
        }

        public double Derivative(double value)
        {
            return Calculate(value) * (1 - Calculate(value));
        }
    }
}
