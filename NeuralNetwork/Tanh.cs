using System;

namespace NeuralNetwork
{
    public class Tanh : IActivation
    {
        public  double Calculate(double value)
        {
            return Math.Tanh(value);
        }

        public  double Derivative(double value)
        {
            return 1 - Math.Pow(Calculate(value), 2);
        }

    }
}
