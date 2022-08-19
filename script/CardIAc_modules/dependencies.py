# ----------------------------------------------------
# Ing. Lucca Dellazoppa - Instituto Balseiro 
# lucca.dellazoppa.m@gmail.com
# ---------------------------------------------------- 
import qt, slicer

def qtMessage(title='',text='',critical_icon_flag = 0):
    msg = qt.QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(text)
    if critical_icon_flag == 1:
      msg.setIcon(qt.QMessageBox.Critical)
    msg.exec_()

def install_tensorflow():
  qm = qt.QMessageBox()
  msg = "It is necessary to install Tensorflow library for 3DSlicer in order to use CardIAc. Do you want to insall it?"
  ret = qm.question(None,'Tensorflow missing', msg, qm.Yes | qm.No)
  if ret == qm.Yes:
    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
    try:
      slicer.util.pip_install('tensorflow')
      slicer.app.restoreOverrideCursor()
      qtMessage("Succes","Tensorflow is now installed!")
    except:
      slicer.app.restoreOverrideCursor()
      text = "Error trying to install tensorflow. Try manual installation: 'slicer.util.pip_install('tensorflow')'."
      qtMessage("Error",text,1)
  if ret == qm.No:
    text = "It won't be possible to segment until tensorflow is installed in 3DSlicer."
    qtMessage("Warning",text,1)

def install_skimage():
  qm = qt.QMessageBox()
  msg = "It is necessary to install Scikit-Image library for 3DSlicer in order to use CardIAc. Do you want to insall it?"
  ret = qm.question(None,'Scikit-Image missing', msg, qm.Yes | qm.No)
  if ret == qm.Yes:
    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
    try:
      slicer.util.pip_install('scikit-image')
      slicer.app.restoreOverrideCursor()
      qtMessage("Succes","Scikit-Image is now installed!")
    except:
      slicer.app.restoreOverrideCursor()
      text = "Error trying to install Scikit-Image. Try manual installation: 'slicer.util.pip_install('skimage')'."
      qtMessage("Error",text,1)
  if ret == qm.No:
    text = "It won't be possible to segment until Scikit-Image is installed in 3DSlicer."
    qtMessage("Warning",text,1)
