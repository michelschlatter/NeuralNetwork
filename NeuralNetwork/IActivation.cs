using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public interface IActivation
    {
         double Calculate(double value);
         double Derivative(double value);
    }
}
