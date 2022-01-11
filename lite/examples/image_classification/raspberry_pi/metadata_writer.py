from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata

ImageClassifierWriter = image_classifier.MetadataWriter
_MODEL_PATH = "eyemodel1.tflite"
_LABEL_FILE = "labelmap.txt"
_SAVE_TO_PATH = "eyemodel1_metadata.tflite"

writer = ImageClassifierWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [127.5], [127.5], [_LABEL_FILE])
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)

# Verify the populated metadata and associated files.
displayer = metadata.MetadataDisplayer.with_model_file(_SAVE_TO_PATH)
print("Metadata populated:")
print(displayer.get_metadata_json())
print("Associated file(s) populated:")
print(displayer.get_packed_associated_file_list())