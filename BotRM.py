# /bin/python3
from aiogram import Bot, types
import asyncio
from aiogram.types import InputFile, ContentType, base, \
KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardMarkup, \
InlineKeyboardButton, CallbackQuery, ReplyKeyboardRemove
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import torch 
import torch.nn as nn
import torchvision.utils as vutils
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import requests
from rembg.bg import remove
import os
import numpy as np





DEVICE = torch.device("cpu")



#инициализация переменных для бота
TOKEN = '5262842962:AAGbHvrgAEltWETA3dratyB-1haMYegmANY'
PATH = 'D:\\!COCO-dataset\\Bot'
USER_PATH = 'D:\\!COCO-dataset\\Bot\\users\\'
bot = Bot(token = TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage = storage)
API = '7d392adf6e00472694b1b29319a7ae11'

#модель для сегментационной сети
model = deeplabv3_resnet101(pretrained=False,num_classes = 1)
model.to(DEVICE)
model.load_state_dict(torch.load('D:\\!COCO-dataset\\Bot\\res101.w', map_location = torch.device('cpu')))

async def segment_image(DIR):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor()
    ])

    test_content_img = Image.open(os.path.join(DIR,'segment.png'))
    ground_width, ground_len = test_content_img.size

    back_transform = torchvision.transforms.Resize((ground_len, ground_width))

    image = transforms(test_content_img)

    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))['out']

    output = torch.where(torch.sigmoid(output) > 0.5, 1, 255)
    output = output.to("cpu")
    output = back_transform(output * image).squeeze(0).numpy()

    output = (output.transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)
    out = Image.fromarray(output)
    out.save(os.path.join(DIR, 'segment.png'))
    path = os.path.join(DIR,'segment.png')
    return path

async def remover_site(DIR):

    response = requests.post(
        'https://api.benzin.io/v1/removeBackground',
        files={'image_file': open(os.path.join(DIR,'segment.png'), 'rb')},
        data={'size': 'full'},
        headers={'X-Api-Key': '7d392adf6e00472694b1b29319a7ae11'},
    )
    if response.status_code == requests.codes.ok:
        with open(os.path.join(DIR,'no-bg.png'), 'wb') as out:
            out.write(response.content)
        path = os.path.join(DIR,'no-bg.png')
        return path
    else:
        print("Error:", response.status_code, response.text)
        return 'Запрос не прошел, надо попробовать немного попозже'


async def rembg(DIR):
    image = Image.open(os.path.join(DIR,'segment.png'))
    result = remove(image)
    result.save(os.path.join(DIR,'result.png'))
    path = os.path.join(DIR,'result.png')
    return path


#инициализация машины состояний бота
class FSMAdmin(StatesGroup):
    menu = State()
    segment = State()
    process = State()
    process_site = State()
    process_rb = State()

#заводим словарь со всем состояниями для возможности вернуться в основное меню из любого состояния
all_states = [FSMAdmin.menu, FSMAdmin.segment, FSMAdmin.process, FSMAdmin.process_site, FSMAdmin.process_rb]
#инициализация клавиатуры главного меню
button_menu_1 = KeyboardButton('Поехали!')
button_menu_2 = KeyboardButton('Другие ребята')
button_menu_3 = KeyboardButton('RemBG')

menu_kb = ReplyKeyboardMarkup(resize_keyboard=True)
menu_kb.add(button_menu_1).add(button_menu_2).add(button_menu_3)

key_b_exit = ReplyKeyboardMarkup(resize_keyboard=True).add(KeyboardButton('Назад'))


async def set_default_commands(dp):
    await dp.bot.set_my_commands([
        types.BotCommand("help", "Помощь"),
        types.BotCommand("restart", "Запустить бота заново"),
       
    ], scope=types.BotCommandScopeDefault())


@dp.message_handler(commands= ['start'])
async def process_start_command(msg: types.Message):
    await bot.send_message(msg.from_user.id,f'Привет, {msg.from_user.first_name}!\nЯ простой бот, который позволит тебе убрать задний фон с твоей фотографии', reply_markup = menu_kb)
    if not os.path.exists(os.path.join(USER_PATH,str(msg.from_user.id))):
        os.mkdir(os.path.join(USER_PATH,str(msg.from_user.id)))
    else:
        pass
    await FSMAdmin.segment.set()

@dp.message_handler(commands= ['restart'], state = all_states)
async def process_start_command(msg: types.Message, state: FSMContext):
    await FSMAdmin.menu.set()
    await msg.reply(f'Что-то пошло не так, но я в строю, все хорошо! Выбирай, что будем дальше делать', reply_markup = menu_kb)
    if not os.path.exists(os.path.join(USER_PATH,str(msg.from_user.id))):
        os.mkdir(os.path.join(USER_PATH,str(msg.from_user.id)))
    else:
        pass

@dp.message_handler(commands=['help'], state = all_states)
async def process_help_command(msg: types.Message):
    await FSMAdmin.menu.set()
    text = 'Я простой бот, который позволит тебе убрать задний фон с твоей фотографии!\n\nЕсли что-то случилось с ботом, введи команду /restart.\nЕсли с ботом совсем все плохо, напиши этому парню: @shirsergey'
    await bot.send_message(msg.from_user.id,text)

@dp.message_handler(text = 'Назад', state = all_states)
async def menu(msg: types.Message, state: FSMContext):
    
    await bot.send_message(msg.from_user.id, text = 'Выбирай любой вариант и я смогу убрать задний фон с твоей фотографии', reply_markup = menu_kb)

    await FSMAdmin.segment.set()

@dp.message_handler(text = 'Поехали!', state = all_states)
async def segment(msg: types.Message, state: FSMContext):
    
    await bot.send_message(msg.from_user.id, text = 'Под этой кнопкой скрывается обученный мною алгоритм для сегментации изображений, основанный на работе сети DeepLabv3_ResNet101. Алгоритм работает не идеально, но показывается сравнительрно неплохие результаты. Пришли мне фотографию и убедись в этом!', reply_markup= key_b_exit)

    await FSMAdmin.process.set()


@dp.message_handler(content_types= [ContentType.PHOTO, ContentType.DOCUMENT], state = FSMAdmin.process)
async def images(msg: types.Message, state: FSMContext):
    if msg.content_type == ContentType.PHOTO:
        await msg.photo[-1].download(destination_file = USER_PATH + str(msg.from_user.id) + '/segment.png')
    if msg.content_type == ContentType.DOCUMENT:
        file_id = msg.document.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, USER_PATH + str(msg.from_user.id) + '/segment.png')
  
    await bot.send_message(msg.from_user.id, text = 'Супер, буквально пара секунд, и твоя фотография будет готова!', reply_markup = ReplyKeyboardRemove())

    DIR = os.path.join(USER_PATH,str(msg.from_user.id))

    segment = await segment_image(DIR)
    segment_file = InputFile(segment)
    await bot.send_photo(msg.from_user.id, segment_file)
    await FSMAdmin.menu.set()
    await bot.send_message(msg.from_user.id, text = 'Было весело',reply_markup = menu_kb)


@dp.message_handler(text = 'Другие ребята', state = all_states)
async def segment(msg: types.Message, state: FSMContext):
    
    await bot.send_message(msg.from_user.id, text = 'Выбранный алгоритм создан крутыми ребятами из Benzin.io, я использую их API, как один из вариантов помочь тебе убрать задний фон. Присылай фотографию, не стесняйся', reply_markup= key_b_exit)

    await FSMAdmin.process_site.set()

@dp.message_handler(content_types= [ContentType.PHOTO, ContentType.DOCUMENT], state = FSMAdmin.process_site)
async def images_site(msg: types.Message, state: FSMContext):
    if msg.content_type == ContentType.PHOTO:
        await msg.photo[-1].download(destination_file = USER_PATH + str(msg.from_user.id) + '/segment.png')
    if msg.content_type == ContentType.DOCUMENT:
        file_id = msg.document.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, USER_PATH + str(msg.from_user.id) + '/segment.png')
  
    await bot.send_message(msg.from_user.id, text = 'Отлично, сейчас будет счастье!', reply_markup = ReplyKeyboardRemove())

    DIR = os.path.join(USER_PATH,str(msg.from_user.id))

    segment = await remover_site(DIR)
    segment_file = InputFile(segment)
    await bot.send_photo(msg.from_user.id, segment_file)
    await FSMAdmin.segment.set()
    await bot.send_message(msg.from_user.id, text = 'Они круто справились',reply_markup = menu_kb)


@dp.message_handler(text = 'RemBG', state = all_states)
async def segment(msg: types.Message, state: FSMContext):
    
    await bot.send_message(msg.from_user.id, text = 'Нажав на данную кнопку ты выбрал работу библиотеки - RemBG. Эта библиотека основана на использовании сети U2-Net. ', reply_markup= key_b_exit)

    await FSMAdmin.process_rb.set()

@dp.message_handler(content_types= [ContentType.PHOTO, ContentType.DOCUMENT], state = FSMAdmin.process_rb)
async def images_site(msg: types.Message, state: FSMContext):
    if msg.content_type == ContentType.PHOTO:
        await msg.photo[-1].download(destination_file = USER_PATH + str(msg.from_user.id) + '/segment.png')
    if msg.content_type == ContentType.DOCUMENT:
        file_id = msg.document.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, USER_PATH + str(msg.from_user.id) + '/segment.png')
  
    await bot.send_message(msg.from_user.id, text = 'Отлично, сейчас заценим!', reply_markup = ReplyKeyboardRemove())

    DIR = os.path.join(USER_PATH,str(msg.from_user.id))

    segment = await rembg(DIR)
    segment_file = InputFile(segment)
    await bot.send_photo(msg.from_user.id, segment_file)
    await FSMAdmin.segment.set()
    await bot.send_message(msg.from_user.id, text = 'Довольно-таки неплохо',reply_markup = menu_kb)

if __name__ == '__main__':
	executor.start_polling(dp)