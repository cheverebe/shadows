import os
import threading

from main_app.angle_finder import AngleFinder
from main_app.main import MainApp


class BatchProcessor(object):
    OUTPUT_BASE_PATH = 'img/out/'
    TEMP_FOLDER_SUB_PATH = 'temp/'
    ANGLE_FILE_EXTENSION = '.jpg'

    def __init__(self):
        self.origin_path = self.get_origin_folder_path()
        self.sequence_names = self.get_sequence_names_list(self.origin_path)

    def run(self):
        for sequence_name in self.sequence_names:
            names = self.get_folder_names_for(sequence_name)
            main_app = MainApp(
                names['source_folder'],
                names['output_folder'],
                angle_file=names['angle_file'],
                angle_finder_folder=names['angle_finder_input_folder'])
            angle_finder = AngleFinder(names['angle_finder_input_folder'], names['angle_file'])
            self.run_thread(angle_finder.run)
            main_app.run()
            angle_finder.end = True

    def get_folder_names_for(self, sequence_name):
        return {'source_folder': self.origin_path + sequence_name + '/',
                'output_folder': self.OUTPUT_BASE_PATH + sequence_name + '/',
                'angle_finder_input_folder': self.OUTPUT_BASE_PATH + self.TEMP_FOLDER_SUB_PATH + sequence_name + '/',
                'angle_file': self.OUTPUT_BASE_PATH + sequence_name + self.ANGLE_FILE_EXTENSION}

    @staticmethod
    def get_sequence_names_list(path):
        return sorted([f for f in os.listdir(path)])

    @staticmethod
    def get_origin_folder_path():
        origin_path = raw_input("Enter origin folder:")
        if not origin_path.endswith('/'):
            origin_path += '/'
        return origin_path

    @staticmethod
    def run_thread(callable_obj):
        try:
            t = threading.Thread(target=callable_obj)
            t.daemon = True  # set thread to daemon ('ok' won't be printed in this case)
            t.start()
        except:
            print "Error: unable to start thread"
BatchProcessor().run()
