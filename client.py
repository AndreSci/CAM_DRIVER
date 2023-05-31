import time
import traceback

import requests
import datetime

from misc.video_thread import create_cams_threads

from misc.ai import AiClass
from misc.utility import SettingsIni
from misc.logger import Logger

from misc import consts

DUPLICATE_NUMBERS = dict()


def count_duplicate_in(number, time_recon):
    """ Функция проверяет номер был ли он распознан в ближайшее время и возвращает True или False """
    global DUPLICATE_NUMBERS

    result = True

    today = datetime.datetime.now()

    if number in DUPLICATE_NUMBERS:
        # Если между дубликатами время прошло больше заданного одобряем отправку запроса
        if (today - DUPLICATE_NUMBERS[number]['date_time']).total_seconds() > consts.CAR_REPEAT:
            DUPLICATE_NUMBERS[number]['date_time'] = today
        else:
            result = False
    else:
        DUPLICATE_NUMBERS[number] = {'date_time': time_recon}

    return result


def duplicate_numbers(recon_numbers: dict) -> list:
    """ Принимает в себя словарь распознанных номеров, проверяет на повторы по времени и изменяет его """

    ret_value = []

    for it in recon_numbers:

        if count_duplicate_in(recon_numbers[it]['number'], recon_numbers[it]['date_time']):
            result = recon_numbers[it].copy()
            result['date_time'] = str(recon_numbers[it]['date_time'].strftime("%Y-%m-%d/%H.%M.%S"))

            ret_value.append(result)

    return ret_value


def client(logger: Logger, settings_ini: SettingsIni):
    """ Клиент ра"""
    set_ini = settings_ini.take_settings()

    logger.add_log(f"SUCCESS\tclient\tКлиент RASPBERRY_CAM начал свою работу")  # log

    # Создаем объект для поиска номера на кадре
    plate_recon = AiClass()

    # Объект всех камер
    cam_list = create_cams_threads(set_ini['CAMERAS'], logger, plate_recon)

    while True:

        numbers = plate_recon.take_recon_numbers()

        if numbers:
            numbers = duplicate_numbers(numbers)
            request_data = {"result": 'SUCCESS', 'desc': '', 'data': numbers}

            if numbers:
                # вместо request пока что принты
                logger.add_log(f"SUCCESS\tclient\tRequest: {numbers}")  # log

                try:
                    # req = requests.get(f'http://{consts.SERVER_HOST}:{consts.SERVER_PORT}/OnHeartBeat',
                    #                    json=request_data, timeout=1)
                    req = requests.get(f'http://{consts.SERVER_HOST}:{consts.SERVER_PORT}/event/number_on_frame',
                                       json=request_data, timeout=1)
                    print(req.json())

                except Exception as ex:
                    print(f"Тайм-аут ошибка: {ex}")
                    # traceback.print_exc()

        time.sleep(2)


def main():

    # Подгружаем данные из settings.ini
    settings = SettingsIni()
    result = settings.create_settings()

    fail_col = '\033[91m'
    # end_c = '\033[0m'

    # Проверка успешности загрузки данных
    if not result["result"]:
        print(f"{fail_col}")
        print(f"Ошибка запуска сервиса - {result['desc']}")
        input()
        raise Exception("Service error")

    # Обьявляем логирование
    logger = Logger(settings)

    # Запуск сервера фласк
    client(logger, settings)


if __name__ == '__main__':
    main()
