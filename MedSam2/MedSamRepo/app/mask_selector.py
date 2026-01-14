import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
# from app.segmentacion_model import mask_video

class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
    def to_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)

from PyQt5.QtGui import QImage
from pydicom import dcmread, pixel_array
class DCIMHelper:
    def dcimToQImage(dcim_path):
        """
        Convert a DICOM file to a QImage.
        
        Parameters:
        -----------
        dcim_path : str
            Path to the DICOM file
            
        Returns:
        --------
        QImage
            The converted image as a QImage object
            
        Raises:
        -------
        Exception
            If the DICOM file cannot be read or converted
        """
        # Read DICOM file
        ds = dcmread(dcim_path)
        
        # Get pixel array
        pixel_array = ds.pixel_array
        
        # Normalize to 0-255 range
        pixel_array = pixel_array.astype(float)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
        pixel_array = (pixel_array * 255).astype(np.uint8)
        
        # Get dimensions
        height, width = pixel_array.shape
        bytes_per_line = width
        
        # Create and return QImage
        q_image = QImage(pixel_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Make a copy to ensure the data persists after the function returns
        return q_image.copy()

class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self.boxes = []  # will store QGraphicsRectItem
        self.drawing = False
        self.start_pos = None
        self.current_rect_item = None

    def load_image(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        self._scene.clear()
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self.boxes = []
        self.current_rect_item = None

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            # Only start if inside pixmap
            if self._pixmap_item and self._pixmap_item.contains(scene_pos):
                self.drawing = True
                self.start_pos = scene_pos
            else:
                self.start_pos = None
        elif event.button() == QtCore.Qt.RightButton:
            # clear boxes
            for rect in self.boxes:
                if isinstance(rect, QtWidgets.QGraphicsRectItem):
                    self._scene.removeItem(rect)
            self.boxes = []
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.start_pos:
            scene_pos = self.mapToScene(event.pos())
            rect = QtCore.QRectF(self.start_pos, scene_pos).normalized()
            if self.current_rect_item:
                self._scene.removeItem(self.current_rect_item)
            pen = QtGui.QPen(QtCore.Qt.red)
            pen.setWidth(2)
            self.current_rect_item = self._scene.addRect(rect, pen)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.drawing and self.start_pos:
            scene_pos = self.mapToScene(event.pos())
            rect = QtCore.QRectF(self.start_pos, scene_pos).normalized()
            pen = QtGui.QPen(QtCore.Qt.red)
            pen.setWidth(2)
            rect_item = self._scene.addRect(rect, pen)
            self.boxes.append(rect_item)


            if self.current_rect_item:
                self._scene.removeItem(self.current_rect_item)
                self.current_rect_item = None
            self.drawing = False
            self.start_pos = None
        super().mouseReleaseEvent(event)

    def get_boxes_image_coords(self):
        result = []
        if not self._pixmap_item:
            return result
        pixmap = self._pixmap_item.pixmap()
        for rect_item in self.boxes:
            r = rect_item.rect()  # in scene coords
            x1 = max(0, min(pixmap.width(), int(r.left())))
            y1 = max(0, min(pixmap.height(), int(r.top())))
            x2 = max(0, min(pixmap.width(), int(r.right())))
            y2 = max(0, min(pixmap.height(), int(r.bottom())))
            result.append(BBox(x1, y1, x2, y2))
        return result

class Annotator(QtWidgets.QWidget):
    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.folder = folder
        # self.frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.jpg')])
        self.frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.png') or f.lower().endswith('.jpg')])
        if not self.frames:
            raise RuntimeError("No JPG frames found")

        self.current_index = 0
        self.annotations = {i: [] for i in range(len(self.frames))}

        self.view = GraphicsView(self)
        btn_prev = QtWidgets.QPushButton("Previous")
        btn_prev.clicked.connect(self.prev_frame)
        btn_next = QtWidgets.QPushButton("Next")
        btn_next.clicked.connect(self.next_frame)
        btn_export = QtWidgets.QPushButton("Export Data")
        btn_export.clicked.connect(self.export_data)
        layout = QtWidgets.QVBoxLayout(self)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(btn_prev)
        hl.addWidget(btn_next)
        hl.addWidget(btn_export)
        layout.addWidget(self.view)
        layout.addLayout(hl)

        self.load_frame(0)

    def load_frame(self, idx):
        self.current_index = idx
        path = self.frames[idx]
        self.view.load_image(path)
        # reload saved boxes
        for b in self.annotations[idx]:
            rect = QtCore.QRectF(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1)
            pen = QtGui.QPen(QtCore.Qt.red)
            pen.setWidth(2)
            rect_item = self.view._scene.addRect(rect, pen)
            self.view.boxes.append(rect_item)

    def save_current_boxes(self):
        bs = self.view.get_boxes_image_coords()
        self.annotations[self.current_index] = bs

    def prev_frame(self):
        if self.current_index > 0:
            self.save_current_boxes()
            self.load_frame(self.current_index - 1)

    def next_frame(self):
        if self.current_index + 1 < len(self.frames):
            self.save_current_boxes()
            self.load_frame(self.current_index + 1)

    def export_data(self):
        # Before exporting, save the current boxes too
        self.save_current_boxes()

        data = []
        for idx, boxes in self.annotations.items():
            for b in boxes:
                data.append({
                    "frame": idx,
                    "box_prompt": (b.x1, b.y1, b.x2, b.y2)
                })

        mask_video(data, self.folder)

        self.close()
        # out_path = os.path.join(self.folder, "box_data.json")
        # with open(out_path, 'w') as f:
        #     json.dump(data, f, indent=2)
        # print("Saved:", out_path)
        # print("Data:", data)

def main():
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) < 2:
        print("Usage: python annotator.py <frames_folder>")
        sys.exit(1)
    folder = sys.argv[1]
    w = Annotator(folder)
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
