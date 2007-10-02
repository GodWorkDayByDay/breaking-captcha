GenericLayer input, hidden, output;
input =  new GenericLayer(50, NULL, &hidden);
hidden = new GenericLayer(50, &input, &output);
output = new GenericLayer(50, &hidden, NULL);