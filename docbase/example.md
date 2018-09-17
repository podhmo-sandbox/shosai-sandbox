#Example notebook
### Markdown cells

This is an example notebook that can be converted with `nbconvert` to different formats. This is an example of a markdown cell.

### LaTeX Equations

Here is an equation:

$$
y = \sin(x)
$$

### Code cells


```python
print("This is a code cell that produces some output")
```

    This is a code cell that produces some output


### Inline figures


```python
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
```




    [<matplotlib.lines.Line2D at 0x1111b2160>]




![png](https://image.docbase.io/uploads/2127710a-2d92-4736-bd3a-23474e32cf24.png)