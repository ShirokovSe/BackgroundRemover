# BackgroundRemover
TelegramBot for removing background. You can chat with the bot: https://t.me/CleanBackBot

This TelegramBot is written with python and aiogram. He is able to remove background from your photo. For removing task bot has 3 different solution!

# The first solution

The first solution is DeebLabv3 Resnet101 neural network is learnt on CocoDataset. During the learning was used bootstrap samples for decreasig overlearning, also was used scheduler for learning rate. The average iou metric for validation was got near 0.95. 

# The second solution

The second solution is using API for the [Benzin.io](https://benzin.io/). These guys made a pretty well system for removing background.
  
  
  
# The third solution
  
The third solution is applying a rembg library. This library is based on the U2-Net neural network. 
  

# Examples

Examples for every solution.

<table align ="center">
  <tr><th>Original</th><th>Benzin.io</th><th>RemBG</th><th>DeepLabv3</th>
  <tr>
    <td>
    <img src="https://github.com/ShirokovSe/BackgroundRemover/blob/main/Examples/galstuc.jpg" width="150">
    </td>
    <td>
    <img src="https://github.com/ShirokovSe/BackgroundRemover/blob/main/Examples/no-bg.png" width="150">
    </td>
   <td>
   <img src="https://github.com/ShirokovSe/BackgroundRemover/blob/main/Examples/rembg.png" width="150"</td>
  </td>
   <td>
   <img src="https://github.com/ShirokovSe/BackgroundRemover/blob/main/Examples/segment.png" width="150"</td>
   </tr>
</table>
  
 Here is examples for every solution.
 
<table align ="center">
  <tr><th>Original</th><th>Benzin.io</th><th>RemBG</th><th>DeepLabv3</th>
  <tr>
    <td>
    <img src="https://github.com/ShirokovSe/BackgroundRemover/blob/main/Examples/original2.jpg" width="150">
    </td>
    <td>
    <img src="https://github.com/ShirokovSe/BackgroundRemover/blob/main/Examples/no-bg_2.png" width="150">
    </td>
   <td>
   <img src="https://github.com/ShirokovSe/BackgroundRemover/blob/main/Examples/result_2.png" width="150"</td>
  </td>
   <td>
   <img src="https://github.com/ShirokovSe/BackgroundRemover/blob/main/Examples/segment_nn.png" width="150"</td>
   </tr>
</table>



