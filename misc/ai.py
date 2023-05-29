import cv2
import threading
import traceback

import misc.consts as consts
import datetime

from ultralytics import YOLO

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)


class DetectNumber:
    """ Класс поиск и распознания номера на кадре """
    def __init__(self):
        self.model_plates = YOLO(consts.PLATES_Y8_MODEL_PATH)
        self.model_number = YOLO(consts.NUMS_Y8_MODEL_PATH)

        self.detections = dict()
        self.cams_frame = dict()

        # Сюда записываем последние распознанные номера
        # {cam1: dict(numbers: ['A100AA77', ], parsed: true/false), }
        self.lock_change_nums = threading.Lock()
        self.recon_numbers = dict()

    def __convert_number(self, frame) -> str:
        """ Распознает уже подготовленный номер """
        frame = cv2.resize(frame, (consts.NUMS_WIDTH_INPUT, consts.NUMS_HEIGHT_INPUT))
        recon_items = self.model_number(frame, verbose=False,
                                           conf=consts.CONFIDENCE_THRESHOLD, save=False, stream=True)
        list_of_x_and_classes = []

        res = ""

        for result in recon_items:

            boxes = result.boxes.cpu().numpy()
            classes = boxes.cls

            for i in range(len(boxes)):
                list_of_x_and_classes.append((boxes.xyxy[i][0], result.names[int(classes[i])]))
                # print(f"class = {result.names[int(classes[i])]}, x={boxes.xyxy[i][0]}")
            list_of_x_and_classes.sort()

            if list_of_x_and_classes:
                for _, cls in list_of_x_and_classes:
                    res += cls

        return res
        # Boxes object for bbox outputs

    def recon_plate(self, frame) -> list:
        """ Перед отправкой в нейронку нужно произвести с ней манипуляции"""
        # run the YOLO model on the frame

        recon_items = self.model_plates(frame, verbose=False, stream=False)

        biggest = list()

        for data in recon_items[0].boxes.data.tolist():
            # extract the confidence (i.e., probability) associated with the detection
            confidence = data[4]

            # filter out weak detections by ensuring the
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence,
            # draw the bounding box on the frame
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

            if not biggest:
                biggest = [xmin, ymin, xmax, ymax]
            elif (biggest[2] - biggest[0]) < (xmax - xmin):
                biggest = [xmin, ymin, xmax, ymax]

        return biggest

    def recon_number(self, cam_name) -> str:
        """ Возвращает номер в виде списка элементов номера """
        res = ""

        if len(self.detections[cam_name]) == 4:
            xmin, ymin, xmax, ymax = self.detections[cam_name]

            # Вырезаем номер из кадра
            crop_img = self.cams_frame[cam_name][ymin:ymax, xmin:xmax]

            if consts.DEBUG_MODE:
                cv2.imshow(f'{cam_name}', crop_img)
                cv2.waitKey(1)

            res = self.__convert_number(crop_img)

        return res


class AiClass(DetectNumber):
    def __init__(self):
        super().__init__()
        self.lock_thread_allow_recon = threading.Lock()
        self.lock_thread_copy_frame = threading.Lock()
        self.allow_recognition_by_name = dict()

        self.allow_rec_lock = threading.Lock()
        self.allow_recognition = dict()

        self.start_recon = dict()

        self.lock_box_rectangle = threading.Lock()

        self.threads_for_recon = dict()
        self.threads_for_recon['one_cam'] = \
            threading.Thread(target=self.__thread_find, daemon=True)
        self.threads_for_recon['one_cam'].start()

    # ФУНКЦИЯ СТАРТ
    def find_plates(self, frame, cam_name: str):
        """ Функция начала распознавания номера в отдельном потоке """

        with self.lock_thread_allow_recon:
            if cam_name not in self.allow_recognition_by_name:
                self.allow_recognition_by_name[cam_name] = True

            allow_recon = self.allow_recognition_by_name[cam_name]

        if allow_recon:

            with self.lock_thread_allow_recon:
                self.allow_recognition_by_name[cam_name] = False

            self.cams_frame[cam_name] = frame.copy()

            with self.allow_rec_lock:
                self.allow_recognition[cam_name] = True

    # Функция для запуска в отдельном потоке для каждой камеры
    def __thread_find(self):

        while True:  # Если нет команды остановить поток

            start_recon = True

            # Проверяем камеры на готовность распознаваться
            with self.allow_rec_lock:
                for key in self.allow_recognition:
                    if not self.allow_recognition[key]:
                        start_recon = False

            if start_recon:  # Если все камеры передали кадр начать распознавание

                with self.allow_rec_lock:
                    for key in self.allow_recognition:
                        self.allow_recognition[key] = False

                # Получаем время для статистики скорости распознавания
                start_time = datetime.datetime.now()

                try:

                    # Решает вопрос с изменением размера в процессе добавления новых камер из другого потока
                    with self.allow_rec_lock:
                        copy_allow = self.allow_recognition.copy()

                    for key in copy_allow:
                        self.detections[key] = self.recon_plate(self.cams_frame[key])

                    # Показываем кадр результат тестов +5%-10% к времени распознания
                    for key in copy_allow:
                        if consts.DEBUG_MODE:
                            self.__img_show(key)

                    for key in copy_allow:
                        number = self.recon_number(key)

                        if len(number) > 0:
                            number = ''.join(number)
                            # TODO реализовать на разные страны

                            if len(number) > consts.LEN_FOR_NUMBER:
                                # Получаем время
                                end_time = datetime.datetime.now()
                                delta_time = (end_time - start_time).total_seconds()

                                with self.lock_change_nums:
                                    self.recon_numbers[key] = {'numbers': [number, ], 'parsed': False,
                                                               'date_time': end_time,
                                                               'recognition_speed': delta_time}

                except Exception as ex:
                    print(f"EXCEPTION\tAiClass.__thread_find\tИсключение в работе распознавания номера: {ex}")
                    traceback.print_exc()

                for key in self.allow_recognition_by_name:
                    self.allow_recognition_by_name[key] = True

    def __img_show(self, cam_name):
        """ Функция выводит кадр в отдельном окне с размеченным номером """
        if consts.DEBUG_MODE:
            with self.lock_box_rectangle:
                if len(self.detections[cam_name]) == 4:
                    xmin, ymin, xmax, ymax = self.detections[cam_name]
                    cv2.rectangle(self.cams_frame[cam_name], (xmin, ymin), (xmax, ymax), GREEN, 2)

            # Показываем кадр с нарисованным квадратом
            cv2.imshow(f'show: {cam_name}', self.cams_frame[cam_name])
            cv2.waitKey(10)

    def take_recon_numbers(self) -> dict:
        """ Получить копию распознанных номеров """
        with self.lock_change_nums:
            copy_rec = dict()

            # Проверяем на новые события
            for it in self.recon_numbers:
                if not self.recon_numbers[it]['parsed']:
                    copy_rec[it] = self.recon_numbers[it].copy()
                    self.recon_numbers[it]['parsed'] = True

            return copy_rec
