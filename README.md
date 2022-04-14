# BackgroundRemover
TelegramBot for removing background

This TelegramBot is written with python and aiogram. He is able to remove background from your photo. For removing task bot has 3 different solution!

# The first solution

The first solution is DeebLabv3 Resnet101 neural network is learnt in CocoDataset. During the learning was used bootstrap samples for decreasig overlearning, also was used scheduler for learning rate.

# The second solution

The second solution is using API for the [Benzin.io](https://benzin.io/). These guys made a pretty well system for removing background.
  
  
  
  # The third solution
  
  The third solution is applying a rembg library. This library is based on the U2-Net neural network. 
  

# Examples
Here is examples for every solution.
<table align ="center">
  <tr><th>Benzin.io</th><th>RemBG</th><th>DeepLabvv3</th>
  <tr>
    <td>
    <img src="https://github.com/ShirokovSe/BackgroundRemover/tree/main/Examples/no_bg.png" width="150">
    </td>
   <td>
   <img src="https://github.com/ShirokovSe/BackgroundRemover/tree/main/Examples/rembg.png" width="150"</td>
  </td>
   <td>
   <img src="https://github.com/ShirokovSe/BackgroundRemover/tree/main/Examples/segment.png" width="150"</td>
   </tr>
</table>
  
 Here is examples for every solution.
 
<table align ="center">
  <tr><th>Original</th><th>Benzin.io</th><th>RemBG</th><th>DeepLabvv3</th>
  <tr>
    <td>
    <img src="https://github.com/ShirokovSe/BackgroundRemover/tree/main/Examples/original2.png" width="150">
    </td>
    <td>
    <img src="https://github.com/ShirokovSe/BackgroundRemover/tree/main/Examples/no_bg_2.png" width="150">
    </td>
   <td>
   <img src="https://github.com/ShirokovSe/BackgroundRemover/tree/main/Examples/result_2.png" width="150"</td>
  </td>
   <td>
   <img src="https://github.com/ShirokovSe/BackgroundRemover/tree/main/Examples/segment_nn.png" width="150"</td>
   </tr>
</table>
As you can see, suggested approach works with a good quality and it takes near 7-8 seconds on CPU to process it.


