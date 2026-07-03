# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:30:45 2022

@authors: Manuel

TODO:
- move the UI into the Tracking class so that you don't have to readdress the MP tools
- TCP server multi-client...
+ multi-animal triangulation and the identification problem:
 (-) reuse last valid combination first, probably works for the next frame (not necessary with nFish==1)
  + test all different combinations and exclude those out of tank
  + go through all combinations to check if two solutions can occur
  - what criteria when two solutions, first? use previous? : FIRST

(-) identification problem:
  - before triangulation (in pos2D):
    - crossing problems with minimal distance heuristic
    - can be used for triangulation: pairs remain the same so need be detected once only
  - after triangulation (in pos3D):
    - reduced crossing problems with minimal distance heuristic
    - use triangulation pair to transfer identification to 2D images

FAILED:
- start cameras at Tracking init and close at destroy
  -> attempt failed, thread writing in wrong memory space
"""

import multiprocessing as mp
import threading
import socket
import time
import sys
import os
import re
import csv
from ctypes import c_wchar_p
import numpy as np
import cv2
from ximea import xiapi
from PyQt5.QtWidgets import *

# IPs and communication ports (warning: ports indexed by camNb not camInd)
IP = {'Tracking': '192.168.0.2', 'Rendering': '192.168.0.1', 'localhost':'127.0.0.1'}
UDPclientRendering = (IP['Rendering'], 50771)
# UDPclientRendering = (IP['localhost'], 50771)     # For local debug
UDPserverRenderingPort = 50772
TCPserverTracking = (IP['Tracking'], 65432)
# TCPserverTracking = (IP['localhost'], 65432)      # For local debug
UDPpacketSize = 65000   # Size of UDP packets
TCPpacketSize = 4096    # Size of TCP packets

# Image global settings (color images)
imgWidth, imgHeight = 1280, 1024

# DLC key variables
nKeysMax = 20       # Maximum number of keys detected by DLC
keyRadius = 3       # Maximum radius size of the keys for monitoring (when inference p=1)
cyclopRadius = 4    # Maximum radius size of cyclop for monitoring (when inference p=1)
cyclopColor = 0     # Detection or cyclop key center color (white)

# UE stimulus event related
nEventsMax = 4     # Maximum number of events that can be sent back to results

# Stores all possible pair combinations, between cameras (triangulation) or from one camera frame to the next (identification)
pairLists = []

# Settings place holder
class Settings:
    owner = None

GUIprocess = None

# Tracking that detects 2D positions of fish for each camera, triangulates and runs DLC on cropped images
class Tracking:

    # Constructor with param initialization
    def __init__(self):
        """Initializes experiment's parameters"""

        # Log object
        self.log = Log(logLevel=2, showTime=True)
        self.log.LogText(1, 'Tracking() called')

        # Load settings and set important parameters
        self.LoadSettings()
        self.camCount = len(self.camList)

        # Global control flags (communication between processes)
        self.startRequest = mp.Value('B', False)             # Start video acquisition when True
        self.expStarted = mp.Value('B', False)               # Experiment started when True (acquiring images and monitoring)
        self.trialStarted = mp.Value('B', False)             # Trial started when True
        self.stopRequest = mp.Value('B', False)              # stop video acquisition when True
        self.quit = mp.Value('B', False)                     # Quit program when True
        self.dataRecording = mp.Value('B', False)            # Start/stop recording data when True/False
        self.serverRunning = mp.Value('B', False)            # TCP server on/off when True/False (used to launch the GUI)
        self.acquiring = mp.Array('B', [False] * self.camCount)        # Acquiring when True (for each camera)
        self.detectRunning = mp.Array('B', [False] * self.camCount)    # Detection running on all cameras (simple or blob detector)
        self.DLCrunning = mp.Value('B', False)                   # DLC inferences running on all cameras
        self.syncRequest = mp.Array('B', [False] * self.camCount)      # Synch request (reset startIndexes) when True (for each camera)
        self.newRefRequest = mp.Array('B', [False] * self.camCount)    # Request new reference images when True (for each camera)
        
        # Manager to share lists of objects across processes and threads
        manager = mp.Manager()
        
        # Images streamed between processes
        self.imgCrops = manager.list([None] * self.camCount)       # 2 cropped images, one per camera
        self.imgMonits = manager.list([None] * self.camCount * 2)  # Monitoring images for upper [0,1] / lower [2,3] panel for camera pairs
        self.imgIndexes = mp.Array('l', [-1] * self.camCount)      # Index of last image received in each camera (stream)
        
        # Detected 2D positions of fishes in each camera (10 to ensure we have enough)
        self.pos2Ds = manager.list([np.zeros((10, 2))] * self.camCount)    # Detected 2D positions of fishes for each camera (in full image coordinates)
        self.nFishDetect2D = mp.Array('H', [0] * self.camCount)            # Number of detected fishes for each camera (2D pos)
        
        # Triangulated 3D position of animal
        self.pos3Ds = manager.list([np.zeros((10, 3))])      # Fish 3D positions triangulated from valid pos2Ds of each camera (in aquarium coordinates)
        self.nFishDetect3D = mp.Value('H', 0)                # Number of detected fishes for each camera (2D pos)
        self.imgIndexPos3D = mp.Value('l', -1)               # Index of cam0 image from which pos2D has been triangulated
        
        # DLC key variables
        self.pos2D_cyclop = manager.list([np.zeros(3)] * self.camCount)         # Computed cyclop 2D positions for each camera (needs DLC activated, in full image coordinates)
        self.pos3D_cyclop = manager.list([np.zeros(4)])                         # Triangulated cyclop 3D positions (X, Y, Z, pMean) (in aquarium coordinates)
        
        self.pos2Ds_DLC = manager.list([np.zeros((nKeysMax, 3))] * self.camCount)   # 2D positions and confidence (X, Y, p) of each inferred key (in full image coordinates)
        self.imgIndexes_DLC = manager.list([0] * self.camCount)                     # Index of image from which keys have been inferred
        
        # Triangulated DLC inferred keys
        self.pos3Ds_DLC = manager.list([np.zeros((nKeysMax, 4))])                   # Triangulated 3D positions (X, Y, Z, pMean) of each inferred pair (in aquarium coordinates)
        self.imgIndexPos3D_DLC = mp.Value('l', -1)              # Index of cam0 image from which DLC keys have been inferred

        # Rendering returned event infos
        self.timeRendering = mp.Value('f', 0.0)
        self.posTracked = manager.list([np.zeros(3)])                       # 3D positions (X, Y, Z) of focal fish returned by rendering PC
        self.posEvents = manager.list([np.zeros(3)] * nEventsMax)          # 3D positions (X, Y, Z) of each event (in aquarium coordinates)
        self.rotEvents = manager.list([np.zeros(3)] * nEventsMax)          # 3D rotations (rX, rY, rZ) of each event (in aquarium coordinates)
        self.nEvents = manager.Value('H', 0)                 # Number of actual events to update

        # Trial infos for results directory and filename of results, and to update UI
        self.expID = manager.Value(c_wchar_p, 'exp')
        self.subjectID = manager.Value(c_wchar_p, 'subj')
        self.trialID = manager.Value(c_wchar_p, 'trial')
        self.condID = manager.Value(c_wchar_p, 'cond')
        self.recVideos = mp.Value('B', True)
        self.saveResults = mp.Value('B', True)
        
        # UI shared controller vars
        self.speciesName = manager.Value(c_wchar_p, self.settings.speciesName)
        self.runDetect = mp.Value('B', self.settings.runDetect)
        self.runDLC = mp.Value('B', self.settings.runDLC)
        self.triangulate = mp.Value('B', self.settings.triangulate)
        self.showPos2D = mp.Value('B', self.settings.showPos2D)
        self.showDLC = mp.Value('B', self.settings.showDLC)
        self.useCyclop = mp.Value('B', self.settings.useCyclop)
        self.sendPos3D = mp.Value('B', self.settings.sendPos3D)
        self.recvEventPos = mp.Value('B', self.settings.recvEventPos)
        self.saveResults = mp.Value('B', self.settings.saveResults)
        self.imgModes = manager.list(self.settings.imgModes)       # Image monitoring modes (2 max among full, crop, thresh, morph depending on detector)

        # Run sanity check for newly loaded settings
        self.CheckSettings(atLoad=True)

        # Initialize images for monitoring
        self.imgCropDim = (self.cropSize[1], self.cropSize[0], 3)
        self.imgCrops[:self.camCount] = [np.zeros(self.imgCropDim, 'uint8')] * self.camCount
        self.imgIndexes[:self.camCount] = [0] * self.camCount

        # Starts TCP server to receive commands from Rendering PC
        TCPserverThread = threading.Thread(target=self.TCPserver, args=())
        TCPserverThread.start()

        # Waits TCP server to be up and running
        while not self.serverRunning.value: pass
        time.sleep(0.1)

        # Starts GUI
        mp.Process(target=self.StartGUI).start()
        # self.GUIprocess = mp.Process(target=self.StartGUI)       # Cannot pickle self if a process is stored in self
        # self.GUIprocess.start()
        self.log.LogText(2, 'Tracking: GUI process started')

        while True:
            # self.log.LogText(2, 'Tracking: %d' % self.startRequest.value)

            if self.startRequest.value:
                self.startRequest.value = False
                if self.expStarted.value:
                    self.log.LogText(2, 'Tracking: start request received but ignored (already started)')
                else:
                    self.log.LogText(2, 'Tracking: start request received')

                    # Starts tracking
                    self.Start()

            if self.stopRequest.value:
                if not self.expStarted.value:
                    self.log.LogText(2, 'Tracking: stop request received but ignored (not started)')
                    time.sleep(0.2)
                    self.stopRequest.value = False
                else:
                    self.log.LogText(2, 'Tracking: stop request received')
                    time.sleep(0.2)

                    # Closes openCV windows if any
                    cv2.destroyAllWindows()

                    # Back to idle status (all unnecessary threads and processes are closed now)
                    self.acquiring[:self.camCount] = [False] * self.camCount
                    self.expStarted.value = False
                    self.trialStarted.value = False
                    self.stopRequest.value = False
                    self.startRequest.value = False
                    self.log.LogText(2, 'Tracking: waiting for start request')

            if self.quit.value:
                self.log.LogText(2, 'Tracking: quit request received')
                break

    def __del__(self):
        """Destructor"""

        # Destructor is called each time a process ends (in spawn mode...)
        self.log.LogText(3, 'Tracking destructor called')

        # Close openCV windows if any
        cv2.destroyAllWindows()

        # Quit GUI if still there
        # self.GUIprocess.kill()

    def LoadSettings(self):
        """Loads settings files"""

        self.log.LogText(1, 'LoadSettings() called')

        # System settings place-holder
        self.settings = Settings()

        # Gets settings
        self.LoadSettingsFile('System settings', storeVariable='self.settings')
        self.LoadSettingsFile('Camera settings')
        # self.LoadSettingsFile('Detection settings - %s' % self.speciesName.value)
        # self.nKeys = len(self.keyNames)  # Number of inferred keys by DLC
        self.log.logLevel = self.settings.logLevel  # Updates log level

    def LoadSettingsFile(self, settingsName, storeVariable='self'):
        """Loads specific settings to self or another object's attributes"""

        filename = os.path.join('Settings', settingsName + '.txt')
        if not os.path.isfile(filename):
            return

        with open(filename, 'r') as fSet:
            settingLines = fSet.readlines()
            for settingLine in settingLines:
                settingArgs = re.split('\t+', settingLine)
                if len(settingArgs) < 2 or settingArgs[0][0] == '#':
                    continue
                exec("{}.{}={}".format(storeVariable, *settingArgs))

        self.log.LogText(2, '\'%s\' loaded' % settingsName)

    def CheckSettings(self, atLoad=True):
        """Run sanity check on loaded settings"""

        self.log.LogText(1, 'CheckSettings() called')

        if atLoad:
            # Loads settings to UI shared controller vars
            self.speciesName.value = self.settings.speciesName
            self.runDetect.value = self.settings.runDetect
            self.runDLC.value = self.settings.runDLC
            self.triangulate.value = self.settings.triangulate
            self.showPos2D.value = self.settings.showPos2D
            self.useCyclop.value = self.settings.useCyclop
            self.showDLC.value = self.settings.showDLC
            self.sendPos3D.value = self.settings.sendPos3D
            self.recvEventPos.value = self.settings.recvEventPos
            self.imgModes[:] = self.settings.imgModes
            self.saveResults.value = self.settings.saveResults

        self.LoadSettingsFile('Detection settings - %s' % self.speciesName.value)
        self.nKeys = len(self.keyNames)  # Number of inferred keys by DLC

        # Update camCount, trial infos and system infos
        self.camCount = len(self.camList)

        if self.runDetect.value:
            # Force simple detector
            self.camCount = len(self.camList)
            self.simpleDetector = (self.forceSimpeDetector and self.nFish == 1)
            if self.simpleDetector:
                self.nBlobsMax = 1

            # Force only one extra blob if nFish==1
            if self.nFish == 1:
                if self.nBlobsMax > 2:
                    self.log.LogText(2, 'CheckSettings: nBlobsMax=%d but the max value with nFish=1 is two, setting nBlobsMax=2' % self.nBlobsMax)
                    self.nBlobsMax = 2

        # If single camera
        if self.camCount == 1:
            if self.triangulate.value:
                self.log.LogText(2, 'CheckSettings: triangulation requested but only one camera is active, ignoring')
                self.triangulate.value = False
                self.sendPos3D.value = False
                self.recvEventPos.value = False

        if self.nFish > 1:
            self.sendPos3D.value = False
            self.recvEventPos.value = False

        if not self.runDLC.value:
            self.useCyclop.value = False

        # Monitoring modes compatibility

        # If full image requested, it must be in mode[0]
        if self.imgModes[1] == 'full':
            self.log.LogText(2, 'CheckSettings: only upper panel can be in mode \'full\', swapping upper/lower')
            self.imgModes[0], self.imgModes[1] = 'full', self.imgModes[0]

        # If full image requested, mode[1] is 'none'
        if self.imgModes[0] == 'full' and self.imgModes[1] != 'none':
            self.log.LogText(2, 'CheckSettings: if upper panel is full, lower panel must be off')
            self.imgModes[1] == 'none'


        # If upper is 'none', forcing 'crop'
        if self.imgModes[0] in ['', 'none']:
            self.log.LogText(2, 'CheckSettings: mode[0]=\'%s\', switching to \'crop\' to enable monitoring' % self.imgModes[0])
            self.imgModes[0] = 'crop'
            self.imgModes[1] = 'none'

        # If blob detector, thresh or morph depend on blobDetectMode
        if self.runDetect.value and not self.simpleDetector and self.blobDetectMode != 'morph':
            for mIndex in range(2):
                if self.blobDetectMode == 'diff':
                    if self.imgModes[mIndex] in ['thresh', 'morph']:
                        self.log.LogText(2, 'CheckSettings: cannot use mode[%d]=\'%s\' with blobDetectMode=\'%s\', switching to \'diff\'' % (mIndex, self.imgModes[mIndex], self.blobDetectMode))
                        self.imgModes[mIndex] = 'diff'
                elif self.blobDetectMode == 'thresh':
                    if self.imgModes[mIndex] == 'morph':
                        self.log.LogText(2, 'CheckSettings: cannot use mode[%d]=\'%s\' with blobDetectMode=\'%s\', switching to \'thresh\'' % (mIndex, self.imgModes[mIndex], self.blobDetectMode))
                        self.imgModes[mIndex] = 'thresh'

        # If no detection
        if not self.runDetect.value and not self.runDLC.value:
            self.log.LogText(2, 'CheckSettings: no detection enabled (runDetect and runDLC are set to False)')
            self.triangulate.value = False
            self.showPos2D.value = False
            self.showDLC.value = False
            self.sendPos3D.value = False
            self.recvEventPos.value = False
            self.imgModes[0] = 'crop'
            self.imgModes[1] = 'none'

        # If running DLC but not classic detection
        if not self.runDetect.value and self.runDLC.value:
            if self.imgModes[0] in ['diff', 'thresh', 'morph']:
                self.log.LogText(2, 'CheckSettings: cannot use mode[0]=\'%s\' with tracking set to false, switching to \'crop\'' % self.imgModes[0])
                self.imgModes[0] = 'crop'
            if self.imgModes[1] in ['diff', 'thresh', 'morph']:
                self.log.LogText(2, 'CheckSettings: cannot use mode[1]=\'%s\' with tracking set to false, removing lower mode' % self.imgModes[1])
                self.imgModes[1] = 'none'

        # If single camera
        if self.camCount == 1:
            if self.triangulate.value:
                self.log.LogText(2, 'CheckSettings: triangulation requested but only one camera is active, ignoring')
                self.triangulate.value = False
                self.sendPos3D.value = False
                self.recvEventPos.value = False

        # If multiple fish
        if self.nFish > 1:
            if self.runDLC.value:
                self.log.LogText(2, 'CheckSettings: DLC inferences requested but more than 1 fish being tracked, ignoring')
                self.runDLC.value = False
                self.showDLC.value = False
            # Prevent use of filters over sliding window (needs to solve identification first)
            self.filter2D = 0
            self.filter3D = 0

    def Start(self):
        """Starts video processing and tracking"""

        self.log.LogText(1, 'Start() called')

        # Dual mode ?
        self.dualMode = self.imgModes[1] not in ['', 'none']

        processList = []

        # Start video processing to get full and cropped images (one per camera)
        videoCaptureProcs = []
        self.acquiring[:self.camCount] = [False] * self.camCount
        for camInd in range(self.camCount):
            videoCaptureProcs.append(mp.Process(target=self.VideoCapture, args=(camInd,)))
        time.sleep(0.1)
        for camInd in range(self.camCount):
            videoCaptureProcs[camInd].start()
        processList.extend(videoCaptureProcs)

        if self.recvEventPos.value:
            # Starts UDP server to receive commands from Rendering PC
            UDPserverThread = threading.Thread(target=self.UDPserver, args=())
            UDPserverThread.start()

        # Waits that all video processing processes are acquiring from cameras
        t0 = time.perf_counter()
        while self.acquiring[:self.camCount] != [True] * self.camCount:
            time.sleep(0.1)
            if (time.perf_counter() - t0 > self.connectTimeout + 1) or self.stopRequest.value:
                self.log.LogText(2, 'Start: could not start all VideoProcessing processes')
                self.stopRequest.value = True
                return -1
        self.expStarted.value = True

        # Send sync request to reset image indexes
        self.log.LogText(2, 'Start: sending syncRequest')
        self.syncRequest[:self.camCount] = [True] * self.camCount
        while self.syncRequest[:self.camCount] != [False] * self.camCount: pass

        # Start detection processes (one per camera)
        runDetectProcs = []
        if self.runDetect.value:
            self.detectRunning[:self.camCount] = [False] * self.camCount
            for camInd in range(self.camCount):
                runDetectProcs.append(mp.Process(target=self.RunDetect, args=(camInd,)))
            time.sleep(0.1)
            for camInd in range(self.camCount):
                runDetectProcs[camInd].start()
            processList.extend(runDetectProcs)
            while self.detectRunning[:self.camCount] != [True] * self.camCount:
                time.sleep(0.1)

        # Start DLC inferences process (which will start 2 threads)
        if self.runDLC.value:
            self.DLCrunning.value = False
            runDLCproc = mp.Process(target=self.RunDLC, args=())
            runDLCproc.start()
            processList.append(runDLCproc)
            while not self.DLCrunning.value:
                time.sleep(0.1)

        if self.runDetect.value or self.runDLC.value:
            # Start triangulation process (only one)
            if self.triangulate.value:
                self.log.LogText(1, 'Start: start triangulation')
                triangulationProc = mp.Process(target=self.Triangulation)       # pCutoff not used (image thresholding only)
                triangulationProc.start()
                processList.append(triangulationProc)
            time.sleep(0.25)

            # Start results saving process, 2D (per camera) and 3D triangulated (detect and DLC)
            saveResultsProc = mp.Process(target=self.SaveResults)
            saveResultsProc.start()
            processList.append(saveResultsProc)
            time.sleep(0.25)

        # Show panel with 2+2 images, with 2D detection and DLC keys if requested
        self.Monitoring()  # Blocking (main thread)

        # Waits until all processes are dead to avoid broken pipes
        self.log.LogText(2, 'Start: waiting for processes to end')
        aliveCount = len(processList)
        while aliveCount != 0:
            aliveCount = 0
            for proc in processList:
                if proc.is_alive():
                    aliveCount += 1
        self.log.LogText(2, 'Start: processes ended')

        # End of acquisition, ends nicely
        cv2.destroyAllWindows()

    # Monitoring camera images and tracking results (detection and DLC)
    def Monitoring(self, showPerfs=True):
        """Monitoring of the experiment tracking (blocking in main thread)"""

        self.log.LogText(1, 'Monitoring() called as main thread')

        windowName = 'Images %s' % ('(with DLC)' if self.runDLC.value else '')
        cv2.namedWindow(windowName, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        imgIndexesPrev = [self.imgIndexes[camInd]-1 for camInd in range(self.camCount)]

        # Prepares target panel
        fullSize = self.imgModes[0] == 'full'
        imgMonitDim0 = list(self.imgMonits[0].shape)
        if self.dualMode:
            imgMonitDim1 = list(self.imgMonits[2].shape)
            convertColor1 = len(imgMonitDim0) < len(imgMonitDim1)
            convertColor2 = len(imgMonitDim1) < len(imgMonitDim0)
        if abs(self.rotateCamList[0]) != 0:
            imgMonitDim0[0], imgMonitDim0[1] = imgMonitDim0[1], imgMonitDim0[0]         # Swap dimensions (for rotations)
            if self.dualMode:
                imgMonitDim1[0], imgMonitDim1[1] = imgMonitDim1[1], imgMonitDim1[0]     # Swap dimensions (for rotations)
        if self.imgModes[0] == 'full':
            imgMonitDim0[0] //= 2
            imgMonitDim0[1] //= 2
        panelDim = []
        if self.dualMode:
            panelDim.append(imgMonitDim0[0] + imgMonitDim1[0])      # Height : two modes one for each row
        else:
            panelDim.append(imgMonitDim0[0])                        # Height : one mode so one row
        panelDim.append(imgMonitDim0[1] * 2)                        # Width : two cameras in a row
        if len(imgMonitDim0) == 3 or len(imgMonitDim1) == 3:
            panelDim.append(3)                            # Color channels if in color mode
        imgPanel = np.zeros(panelDim, 'uint8')
        textColor = [[255] * 3 if self.imgModes[0] in ['diff', 'thresh', 'morph'] else 0]
        if self.dualMode:
            textColor.append([255]*3 if self.imgModes[1] in ['diff', 'thresh', 'morph'] else 0)

        # Initializes performance profiler
        if showPerfs:
            tPerf = time.perf_counter()
            missed = 0
            missedCount = 0
            updatedCamCount = 0
            perfStr = ''
            dtPerf = (1 // (1/self.framerate)) * (1/self.framerate)

        while True:

            # Quits on stopAcquisition
            if self.stopRequest.value:
                cv2.destroyAllWindows()
                self.log.LogText(1, 'Monitoring: stop request received, quitting')
                return

            # Process new set of images
            sameCounter = 0
            for camInd in range(self.camCount):

                # Continues if same image
                if self.imgIndexes[camInd] == imgIndexesPrev[camInd]:
                    sameCounter += 1
                    continue

                # Update performance counter
                if showPerfs:
                    missed += self.imgIndexes[camInd] - imgIndexesPrev[camInd] - 1  # Missed frames
                    updatedCamCount += 1
                imgIndexesPrev[camInd] = self.imgIndexes[camInd]

                # Get last image in the desired format
                # camNb = self.camList[camInd]
                # self.log.LogText(4, 'Monitoring: getting camera %d image %d' % (camNb, self.imgIndexes[camInd]))
                img = np.copy(self.imgMonits[camInd])   # Get monitoring image

                # Draws crop rectangle if full image
                if fullSize:
                    img = cv2.rectangle(img, self.cropULs[camInd], self.cropULs[camInd] + np.array(self.cropSize), textColor, thickness=2)

                if self.showPos2D.value:
                    # self.log.LogText(4, 'Monitoring: add %d detected fish position(s)' % self.nFishDetect2D[0])

                    if self.runDLC.value and self.useCyclop.value:
                        # Draws circle on cyclop position
                        pPos = np.sqrt(self.pos2D_cyclop[camInd][2])      # Inference probability
                        if pPos > 0:
                            if fullSize:
                                xPos, yPos = self.pos2D_cyclop[camInd][:2].astype(int)
                            else:
                                xPos, yPos = self.pos2D_cyclop[camInd][:2].astype(int) - self.cropULs[camInd]
                            img = cv2.circle(img, (xPos, yPos), int(cyclopRadius * pPos), [255-cyclopColor]*3, thickness=cv2.FILLED)
                            img = cv2.circle(img, (xPos, yPos), int(keyRadius * pPos), [cyclopColor]*3, thickness=cv2.FILLED)

                    elif self.runDetect.value:
                        # Draws circle(s) on detected 2D position(s)
                        for fishIndex in range(self.nFishDetect2D[camInd]):
                            if fullSize:
                                xPos, yPos = self.pos2Ds[camInd][fishIndex].astype(int)
                            else:
                                xPos, yPos = self.pos2Ds[camInd][fishIndex].astype(int) - self.cropULs[camInd]
                            img = cv2.circle(img, (xPos, yPos), 5, [255-cyclopColor]*3, thickness=cv2.FILLED)
                            img = cv2.circle(img, (xPos, yPos), 3, [cyclopColor]*3, thickness=cv2.FILLED)

                if self.showDLC.value:
                    # Adds keys inferred by DLC (with lag when shown in real-time)
                    # self.log.LogText(4, 'Monitoring: add keys inferred by DLC on cam%d' % camNb)

                    # try:
                    posKey = np.copy(self.pos2Ds_DLC[camInd])  # Get DLC detected pos2D

                    for keyInd, keyName in enumerate(self.keyNames):
                        pPos = np.sqrt(np.sqrt(posKey[keyInd, 2]))
                        if pPos > 0:
                            # Draws circle on inferred key (size depends on probability)
                            if fullSize:
                                xKey, yKey = posKey[keyInd, :2].astype(int)
                            else:
                                xKey, yKey = posKey[keyInd, :2].astype(int) - self.cropULs[camInd]
                            rKey = int(keyRadius * np.sqrt(posKey[keyInd, 2]))     # Circles with radius depending on inference confidence
                            # self.log.LogText(4, 'Monitoring: on cam%d, key [%s] is drawn at (%d, %d) with r=%d' % (camNb, keyName, xKey, yKey, rKey))
                            img = cv2.circle(img, (xKey, yKey), rKey, self.keyColors[keyInd], thickness=cv2.FILLED)
                    # # except RuntimeWarning as warn:
                    # except ValueError as warn:
                    #     self.log.LogText(1, 'Monitoring: Warning:%s with # %s #' % (warn, self.pos2Ds_DLC[camInd]))

                # Resize images when full
                if fullSize:
                    img = cv2.resize(img, (imgWidth // 2, imgHeight // 2))

                # Rotate image to align monitoring with setup
                if self.rotateCamList[camInd] == 90:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotateCamList[camInd] == -90:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self.rotateCamList[camInd] == 180:
                    img = cv2.rotate(img, cv2.ROTATE_180)

                if self.dualMode:
                    if convertColor1:
                        img = cv2.merge((img, img, img))

                # Add mode
                if camInd == 1:
                    img = cv2.putText(img, 'Mode=\'%s\'' % self.imgModes[0], (imgMonitDim0[1] - 120, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=textColor[0])

                # Add camera detection infos
                if self.triangulate.value and self.nFish == 1:
                    xText = 10 if camInd == 0 else imgMonitDim0[1]-200
                    img = cv2.putText(img, 'Cam%d (%d fish detected)' % (camInd, self.nFishDetect2D[camInd]), (xText, imgMonitDim0[0]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=textColor[0])

                # Places updated image in 2-images upper panel (for monitoring)
                imgPanel[:imgMonitDim0[0], camInd*imgMonitDim0[1]:(camInd+1)*imgMonitDim0[1]] = img

                # Lower panel if dual monitoring mode
                if self.dualMode:
                    imgMonitLow = np.copy(self.imgMonits[camInd+2])

                    # Rotate image to align monitoring with setup
                    if self.rotateCamList[camInd] == 90:
                        imgMonitLow = cv2.rotate(imgMonitLow, cv2.ROTATE_90_CLOCKWISE)
                    elif self.rotateCamList[camInd] == -90:
                        imgMonitLow = cv2.rotate(imgMonitLow, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif self.rotateCamList[camInd] == 180:
                        imgMonitLow = cv2.rotate(imgMonitLow, cv2.ROTATE_180)

                    if convertColor2:
                        imgMonitLow = cv2.merge((imgMonitLow, imgMonitLow, imgMonitLow))

                    # Add mode
                    if camInd == 1:
                        imgMonitLow = cv2.putText(imgMonitLow, 'Mode=\'%s\'' % self.imgModes[1], (imgMonitDim1[1] - 120, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=textColor[1])
                    startIndW = camInd*imgMonitDim0[1] + (imgMonitDim0[1]-imgMonitDim1[1]) // 2
                    # Places updated image in 2-images lower panel (for monitoring)
                    imgPanel[imgMonitDim0[0]:imgMonitDim0[0] + imgMonitDim1[0], startIndW:startIndW + imgMonitDim1[1]] = imgMonitLow

            # If none of the images changed, continues
            if sameCounter == self.camCount:
                continue

            # Compute framerate
            if showPerfs:
                missed /= updatedCamCount
                t1 = time.perf_counter()
                missedCount += missed
                if t1 - tPerf > dtPerf:
                    # Every one second, estimates processed frames
                    percent = 100 * (self.framerate - missedCount) / self.framerate
                    perfStr = 'Processing at %.1ffps (%.1f%%)' % (self.framerate - missedCount, percent)
                    # perfStr = 'Performance: processed %.1f/%.1f frames (%.1f%%)' % (self.framerate - missedCount, self.framerate, percent)
                    missedCount = 0
                    tPerf += dtPerf
                missed = 0
                updatedCamCount = 0
                imgPanel = cv2.putText(imgPanel, perfStr, (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=textColor[0])

            # Add retained triangulation value
            if self.triangulate.value:
                tpos3D = tuple(self.pos3D_cyclop[0][:3]) if self.useCyclop.value else tuple(self.pos3Ds[0][0, :])
                imgPanel = cv2.putText(imgPanel, 'Fish position (%.1f, %.1f, %.1f)' % tpos3D, (imgMonitDim0[1]-100, imgMonitDim0[0]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=textColor[0])

            # Shows image panel
            cv2.imshow(windowName, imgPanel)
            cv2.moveWindow(windowName, 285, 16)      # TODO: move window but allow user to move it
            cv2.waitKey(1)

    # TCP server receiving commands from Rendering PC (THREAD)
    def TCPserver(self, cmdSeparator='\t'):
        """TCP server receiving commands from Rendering PC and GUI(THREAD)"""

        self.log.LogText(1, 'TCPserver() thread running (waiting for Rendering commands)')

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serverSocket:

            # Binds TCP server to the correct port
            while True:
                try:
                    serverSocket.bind(TCPserverTracking)
                except socket.error as errorMsg:
                    self.log.LogText(2, 'TCPserver: could not bind with server address, retrying in 1s')
                    time.sleep(1)
                    continue
                break

            self.log.LogText(2, 'TCPserver: up and running waiting for connections')
            self.serverRunning.value = True         # Creates GUI once TCP server is running

            while True:

                # Waits for client to connect
                self.log.LogText(2, 'TCPserver: listening on %s...' % str(TCPserverTracking))
                serverSocket.listen()
                try:
                    clientSocket, clientAddress = serverSocket.accept()
                except socket.error as errorMsg:
                    self.log.LogText(2, 'TCPserver: connection error: %s' % errorMsg)
                    serverSocket.close()
                    self.quit.value = True
                    return 0
                self.log.LogText(2, 'TCPserver: connected to client %s' % str(clientAddress))

                with clientSocket:

                    # Reads message (one chunks)
                    try:
                        msg = clientSocket.recv(TCPpacketSize)      # Reads next chunk
                    except socket.error as errorMsg:
                        self.log.LogText(2, 'TCPserver: reception error: %s, ignoring' % errorMsg)
                        msg = b''
                    if not msg:         # Empty message: end of transmission
                        clientSocket.close()  # Close client connection
                        self.log.LogText(2, 'TCPserver: closing client connection')
                        continue

                    # Acknowledge reception
                    # clientSocket.sendall(('TCPserver on Tracking: message received').encode())

                    # Reads and processes list of commands
                    cmdList = msg.decode('utf-8').split(cmdSeparator)
                    for cmd in cmdList:
                        if not len(cmd): break

                        self.log.LogText(2, 'TCPserver: command=\'%s\'' % cmd)
                        # Process start, stop and quit commands (dispatched to other threads)
                        if cmd == 'startExperiment':
                            self.log.LogText(2, 'TCPserver: \'startExperiment\' received from rendering')
                            self.CheckSettings(atLoad=False)        # Check if current settings are ok
                            self.dataRecording.value = False
                            self.startRequest.value = True
                        elif cmd == 'endExperiment':
                            self.log.LogText(2, 'TCPserver: \'endExperiment\' received from rendering')
                            self.dataRecording.value = False
                            self.stopRequest.value = True
                        elif cmd == 'startTrial':
                            if len(self.expID.value) > 0 and len(self.subjectID.value) > 0 and len(self.trialID.value) > 0 and len(self.condID.value) > 0:
                                self.log.LogText(2, 'TCPserver: \'startTrial\' received from rendering')
                                self.log.LogText(2, '  with expID=%s, subjectID=%s, trialID=%s, condID=%s (saveResults=%s, recVideos=%s)' % (self.expID.value, self.subjectID.value, self.trialID.value, self.condID.value, self.saveResults.value, self.recVideos.value))
                                # Sync image indexes between cameras
                                self.syncRequest[:self.camCount] = [True] * self.camCount
                                while self.syncRequest[:self.camCount] != [False] * self.camCount: pass
                                time.sleep(0.05)
                                # Start recording data
                                if self.saveResults.value:
                                    self.dataRecording.value = True
                                self.trialStarted.value = True
                                # eventString[0] = 'startTrial'
                            else:
                                self.log.LogText(2, 'TCPserver: \'startTrial\' received but ignored, send expID, subjectID, trialID, condID first')
                        elif cmd == 'endTrial':
                            self.log.LogText(2, 'TCPserver: \'endTrial\' received from rendering')
                            # eventString[0] = 'endTrial'
                            self.dataRecording.value = False
                            self.trialStarted.value = False
                        elif cmd == 'quit':
                            self.log.LogText(2, 'TCPserver: \'quit\' received')
                            self.quit.value = True
                            self.stopRequest.value = True
                            clientSocket.close()
                            serverSocket.close()
                            return 0

                        # From here on parameters cannot be set if trial started ongoing
                        elif self.trialStarted.value:
                            self.log.LogText(3, 'TCPserver: trial started, command \'%s\' will be ignored' % cmd)

                        # Ask for new reference images
                        elif cmd == 'newRef':
                            self.log.LogText(2, 'TCPserver: \'newRef\' received from rendering')
                            self.newRefRequest[:self.camCount] = [True] * self.camCount

                        # Reload settings
                        elif cmd == 'loadSettings':
                            self.log.LogText(2, 'TCPserver: \'loadSettings\' received from rendering')
                            self.LoadSettings()
                            self.CheckSettings(atLoad=True)   # Run sanity check for newly loaded settings

                        # Loads experiment parameters
                        elif 'expID' in cmd:
                            val = cmd.split(sep='=')[1]
                            self.log.LogText(2, 'TCPserver: self.expID=\'%s\'' % val)
                            exec('self.expID.value=\'%s\'' % val)
                        elif 'subjectID' in cmd:
                            val = cmd.split(sep='=')[1]
                            self.log.LogText(2, 'TCPserver: self.subjectID=\'%s\'' % val)
                            exec('self.subjectID.value=\'%s\'' % val)
                        elif 'trialID' in cmd:
                            val = cmd.split(sep='=')[1]
                            self.log.LogText(2, 'TCPserver: self.trialID=\'%s\'' % val)
                            exec('self.trialID.value=\'%s\'' % val)
                        elif 'condID' in cmd:
                            val = cmd.split(sep='=')[1]
                            self.log.LogText(2, 'TCPserver: self.condID=\'%s\'' % val)
                            exec('self.condID.value=\'%s\'' % val)
                        elif 'recVideos' in cmd:
                            val = cmd.split(sep='=')[1]
                            self.log.LogText(2, 'TCPserver: self.recVideos=\'%s\'' % val)
                            exec('self.recVideos.value=%s' % val)
                            # exec('self.recVideos.value=%d' % 1 if (val == 'true') else 0)      # True in UE is true:::
                            # exec('self.recVideos.value=%s' % (val == 'true'))      # True in UE is true:::
                            self.log.LogText(2, 'TCPserver: self.recVideos=%s' % self.recVideos.value)

                        # Unknown command
                        else:
                            self.log.LogText(3, 'TCPserver: command not understood \'%s\'' % cmd)

    # Ximea camera video capture (PROCESS)
    def VideoCapture(self, camInd, showPerfs=False):
        """For each camera, acquires image, streams to manager and adds image to video (PROCESS)"""

        camNb = self.camList[camInd]
        self.log.LogText(1, 'VideoCapture(%d) process running (PID: %d)' % (camNb, mp.current_process().pid))

        nframePrev = -1             # Frame number returned by XiAPI
        indexStart = -1             # To 'softly' synchronize image indexes
        dataRecordingPrev = False   # Stores previous flag state
        video = None                # Stores video writer
        self.acquiring[camInd] = False

        # Initialize Ximea camera and starts data acquisition
        if True:
            # Get camera from Ximea API
            xiCam = xiapi.Camera()
            self.log.LogText(2, 'VideoCapture(%d): Opening camera SN=%s' % (camNb, self.camSerialNbs[camNb]))

            t0 = time.perf_counter()
            while True:
                try:
                    xiCam.open_device_by_SN(self.camSerialNbs[camNb])
                    break
                except Exception as errMsg:
                    self.log.LogText(2, 'VideoCapture(%d): could not open camera (%s), trying later' % (camNb, errMsg))
                if (time.perf_counter() - t0 > self.connectTimeout-1) or self.stopRequest.value:
                    self.log.LogText(1, 'VideoCapture(%d): could not open camera, quitting' % camNb)
                    self.stopRequest.value = True
                    return -1

            # Set camera parameters
            xiCam.set_exposure(self.exposure)
            xiCam.set_gain(self.gain)
            xiCam.set_param('recent_frame', 0)
            xiCam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
            xiCam.set_framerate(self.framerate)
            xiCam.enable_auto_wb()
            # xiCam.set_acq_transport_buffer_commit(32)
            # xiCam.set_acq_buffer_size(xiCam.get_acq_buffer_size_maximum())
            xiCam.set_imgdataformat('XI_RGB24')
            self.log.LogText(2, 'VideoCapture(%d): xiCam params loaded' % camNb)

            # Prepare ROI crop ranges
            Xmin = self.cropULs[camInd][0]
            Xmax = Xmin + self.cropSize[0]
            Ymin = self.cropULs[camInd][1]
            Ymax = Ymin + self.cropSize[1]

            # Create instance of Image to store image data and metadata
            xiImg = xiapi.Image()

            # Start data acquisition
            xiCam.start_acquisition()
            self.log.LogText(2, 'VideoCapture(%d): xiCam data acquisition started' % camNb)

        # Profiler inits
        if showPerfs:
            tm = t0 = time.perf_counter()
            missed = 0
            missedCount = 0

        self.acquiring[camInd] = True
        while True:

            # Quits on stopRequest
            if self.stopRequest.value or self.quit.value:
                self.log.LogText(1, 'VideoCapture(%d): %s received, ending process' % (camNb, 'stop request' if self.stopRequest.value else 'quit'))
                self.acquiring[camInd] = False
                self.dataRecording.value = False     # Sends signal to stop data recording (in case)
                if not dataRecordingPrev:
                    # Quits when no longer recording
                    cv2.destroyAllWindows()
                    if xiCam is not None:       # Closes camera if opened
                        xiCam.stop_acquisition()
                        xiCam.close_device()
                        self.log.LogText(2, 'VideoCapture(%d): camera closed, quitting' % camNb)
                        self.acquiring[camInd] = False
                    return

            # At trial start: create video writers for each camera
            if self.dataRecording.value and not dataRecordingPrev:
                # Creates paths if not existing
                path = self.settings.resultsDir + '%s/%s/' % (self.expID.value, self.subjectID.value)
                resultsFile = 'Trial-%s_Cond-%s' % (self.trialID.value, self.condID.value)
                try:
                    os.makedirs(path)
                except FileExistsError:
                    pass
                if self.recVideos.value:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video = cv2.VideoWriter(path + resultsFile + '_cam%d' % camNb + '.mp4', fourcc, self.framerate, (imgWidth, imgHeight), isColor=True)
                    self.log.LogText(2, ' VideoCapture(%d): video writers created' % camNb)
                else:
                    self.log.LogText(2, ' VideoCapture(%d): videos will not be recorded, as requested' % camNb)
                dataRecordingPrev = True

            # Get data and pass them from camera to img
            xiCam.get_image(xiImg)

            # Skip if same image
            if xiImg.nframe == nframePrev:
                continue

            # Store newly acquired image index
            if self.syncRequest[camInd] or indexStart == -1:
                indexStart = xiImg.nframe
                self.syncRequest[camInd] = False
            imgIndex = xiImg.nframe - indexStart

            # Show performance (missed frames each second and framerate)
            if showPerfs:
                missed += xiImg.nframe - nframePrev - 1     # Missed frames
                t1 = time.perf_counter()
                missedCount += missed
                if t1 - tm > 1:   # Every one second, estimates missed frames
                    missedPercent = 100 * missedCount / self.framerate
                    self.log.LogText(3, 'VideoCapture(%d): missed=%.f/%.1f frames (%.1f%%)' % (camNb, missedCount, self.framerate, missedPercent))
                    missedCount = 0
                    tm = t1
                fps = 1.0/(t1-t0)
                t0 = t1
                missed = 0
                self.log.LogText(3, 'VideoCapture(%d): (%d) framerate=%.1f fps' % (camNb, imgIndex, fps))
            nframePrev = xiImg.nframe

            # Stores newly acquired image numpy array
            imgFull = xiImg.get_image_data_numpy()
            self.log.LogText(3, 'VideoCapture(%d): new image acquired %03d' % (camNb, imgIndex))

            # Prepare cropped image
            imgCrop = np.copy(imgFull[Ymin:Ymax, Xmin:Xmax, :])

            # During trial: update corresponding video
            if self.dataRecording.value and dataRecordingPrev:
                if self.recVideos.value:
                    video.write(imgFull)

            # At trial end: closes video writers and saves
            elif dataRecordingPrev:
                if self.recVideos.value:
                    video.release()
                    video = None
                    self.log.LogText(2, 'VideoCapture(%d): cam%d video closed and saved' % (camNb, camNb))
                dataRecordingPrev = False

            if self.runDLC.value or self.imgModes[0] == 'crop' or self.imgModes[1] == 'crop':
                # Prepare cropped image
                self.imgCrops[camInd] = imgCrop
                self.log.LogText(3, 'VideoCapture(%d): image %03d cropped, upper-left corner (%d, %d)' % (camNb, imgIndex, Xmin, Ymin))

            # Stores into shared variables
            self.imgIndexes[camInd] = imgIndex

            # Stores monitoring images
            if self.imgModes[0] == 'full':
                self.imgMonits[camInd] = imgFull.copy()
            elif self.imgModes[0] == 'crop':
                self.imgMonits[camInd] = imgCrop.copy()
            if self.dualMode:
                if self.imgModes[1] == 'crop':
                    self.imgMonits[camInd+2] = imgCrop.copy()

    # Run detection 2D on cropped image, simple or blob detector (PROCESS)
    def RunDetect(self, camInd, showPerfs=False):
        """Process acquired images from each camera (PROCESS)"""

        camNb = self.camList[camInd]
        self.log.LogText(1, 'RunDetect(%d) process running (PID: %d)' % (camNb, mp.current_process().pid))

        self.detectRunning[camInd] = False
        dataRecordingPrev = False           # Stores previous flag state
        imgIndexPrev = -1                 # Stores previous img index per camera

        # 2D detection inits
        if True:
            pos2D = np.zeros((10, 2))  # Default is UL corner (no detection)

            sortBlobsBySize = False
            removeMirrorBlobs = True

            # Reference image Average placeholder
            imgRefAvg = np.zeros((self.cropSize[1], self.cropSize[0], 3))
            imgRefCount = 0

            # Prepares blob detector parameters
            if not self.simpleDetector:
                params = cv2.SimpleBlobDetector_Params()
                params.filterByArea = self.filterByArea
                params.thresholdStep = self.thresholdStep
                params.minThreshold = self.minThreshold
                params.maxThreshold = self.maxThreshold
                params.minRepeatability = self.minRepeatability
                params.minDistBetweenBlobs = self.minDistBetweenBlobs
                params.filterByColor = self.filterByColor
                params.blobColor = self.blobColor
                params.filterByArea = self.filterByArea
                params.minArea = self.minArea
                params.maxArea = self.maxArea
                params.filterByConvexity = self.filterByCircularity
                params.minCircularity = self.minCircularity
                params.maxCircularity = self.maxCircularity
                params.filterByInertia = self.filterByInertia
                params.minInertiaRatio = self.minInertiaRatio
                params.maxInertiaRatio = self.maxInertiaRatio
                params.filterByConvexity = self.filterByConvexity
                params.minConvexity = self.minConvexity
                params.maxConvexity = self.maxConvexity
                blobDetector = cv2.SimpleBlobDetector_create(params)

            # Load reference image of this camera
            filename = 'References/refCam%d.png' % camNb
            imgRefGray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            # Checks if shape is ok
            if imgRefGray.shape != (self.cropSize[1], self.cropSize[0]):
                self.log.LogText(2, 'RunDetect(%d): \'%s\' reference image has bad shape, forcing a new one' % (camNb, filename))
                self.newRefRequest[camInd] = True
            else:
                self.log.LogText(2, 'RunDetect(%d): \'%s\' reference image loaded' % (camNb, filename))

            # Initialize pos2D filter (average over sliding window)
            posXslide = [self.cropSize[0]//2] * self.filter2Dsize
            posYslide = [self.cropSize[1]//2] * self.filter2Dsize

            # Initialize kernel mask
            kernel = np.ones((int(self.maskSize[0]), int(self.maskSize[1])), np.uint8)
            self.log.LogText(2, 'RunDetect(%d): morphing kernel and 2D filter initialized' % camNb)

            # Local functions and parameters for detecting mirror blobs (fish length ~35 px)
            if removeMirrorBlobs:
                maxBorderDist = 120     # Less than maxBorderDist px from vertical/horizontal border (default: 120)
                maxDist = 120           # Less than maxDist px within the detected pair (default: 100)
                maxOffset = 10          # Less than maxOffset px vertical/horizontal offset (default: 10)

                def dist(blobList, bInd1, bInd2):
                    return ((blobList[bInd1][0] - blobList[bInd2][0]) ** 2 + (blobList[bInd1][1] - blobList[bInd2][1]) ** 2) ** 0.5

                def borderV(blobList, bInd):
                    return abs(self.cropSize[0]/2 - blobList[bInd][0]) > self.cropSize[0]/2 - maxBorderDist  # Less than maxBorderDist px from vertical border

                def borderH(blobList, bInd):
                    return abs(self.cropSize[1]/2 - blobList[bInd][1]) > self.cropSize[0]/2 - maxBorderDist  # Less than maxBorderDist px from horizontal border

                def alignV(blobList, bInd1, bInd2):
                    return abs(blobList[bInd1][0] - blobList[bInd2][0]) < maxOffset  # Less than maxOffset px vertical offset

                def alignH(blobList, bInd1, bInd2):
                    return abs(blobList[bInd1][1] - blobList[bInd2][1]) < maxOffset  # Less than maxOffset px horizontal offset

                def closerV(blobList, bInd1, bInd2):
                    return bInd1 if abs(blobList[bInd1][0]-self.cropSize[0]/2) > abs(blobList[bInd2][0]-self.cropSize[0]/2) else bInd2

                def closerH(blobList, bInd1, bInd2):
                    return bInd1 if abs(blobList[bInd1][1]-self.cropSize[0]/2) > abs(blobList[bInd2][1]-self.cropSize[1]/2) else bInd2

        # Initializes profiler
        if showPerfs:

            tm = t0 = time.perf_counter()
            missed = 0
            missedCount = 0

        self.log.LogText(2, 'RunDetect: processing started')

        self.detectRunning[camInd] = True
        while True:

            # Quits on stopRequest
            if self.stopRequest.value or self.quit.value:
                self.log.LogText(1, 'RunDetect(%d): %s received, process ending' % (camNb, 'stop request' if self.stopRequest.value else 'quit'))
                self.acquiring[camInd] = False
                self.dataRecording.value = False     # Sends signal to stop data recording (in case)
                if not dataRecordingPrev:
                    # Quits when no longer recording
                    cv2.destroyAllWindows()
                    self.detectRunning[camInd] = False
                    return

            # Get img index and skips if same
            imgIndex = self.imgIndexes[camInd]
            if imgIndex == imgIndexPrev:
                continue

            # Get current cropped image
            imgCrop = np.copy(self.imgCrops[camInd])

            # Show performance (missed frames each second and framerate)
            if showPerfs:
                missed += self.imgIndexes[camInd] - imgIndexPrev - 1     # Missed frames
                t1 = time.perf_counter()
                missedCount += missed
                if t1 - tm > 1:   # Every one second, estimates missed frames
                    missedPercent = 100 * missedCount / self.framerate
                    self.log.LogText(3, 'RunDetect: missed=%.f/%.1f frames (%.1f%%)' % (missedCount, self.framerate, missedPercent))
                    missedCount = 0
                    tm = t1
                fps = 1.0/(t1-t0)
                t0 = t1
                missed = 0
                self.log.LogText(3, 'RunDetect: (%d) framerate=%.1f fps' % (imgIndex, fps))

            # Process new reference request
            if self.newRefRequest[camInd]:
                if imgRefCount == 0:
                    # Start
                    self.log.LogText(2, 'RunDetect(%d): starting image sampling for new reference image' % camNb)
                    frameCount = 0
                if frameCount % self.refFrameSample == 0:
                    # Add image to average
                    imgRefAvg = imgRefAvg + imgCrop
                    imgRefCount += 1
                    self.log.LogText(2, 'RunDetect(%d): adding image %d' % (camNb, imgRefCount))
                frameCount += 1
                if imgRefCount == self.refNframes:
                    # Sampling finished, computes average
                    imgRefAvg = imgRefAvg / self.refNframes
                    imgRefAvg = imgRefAvg.astype('uint8')
                    # Converts to grayscale (tracking is off so not done before) and update file for this camera
                    imgRefGray = cv2.cvtColor(imgRefAvg, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('References/refCam%d.png' % camNb, imgRefGray)
                    self.log.LogText(2, 'RunDetect(%d): new reference image updated and stored (%d images averaged)' % (camNb, self.refNframes))
                    # Resets variables
                    imgRefAvg = np.zeros((self.cropSize[1], self.cropSize[0], 3))
                    imgRefCount = 0
                    self.newRefRequest[camInd] = False

            # Prepares grayscale image
            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

            # Subtracts the reference image (grayscale)
            if self.subtractMethod == 0:        # absolute diff (all cases)
                imgDiff = np.abs(imgGray.astype('int16') - imgRefGray.astype('int16')).astype('uint8')
            elif self.subtractMethod == 1:      # img-ref (lighter fish)
                imgDiff = cv2.subtract(imgGray, imgRefGray)
            elif self.subtractMethod == 2:      # ref-img (darker fish)
                imgDiff = cv2.subtract(imgRefGray, imgGray)

            # Detects 2D position of fish in image
            if self.simpleDetector:
                # Thresholds the diff image (binary)
                _, imgThresh = cv2.threshold(imgDiff, self.camThresh, 255, cv2.THRESH_BINARY)

                # Morphing
                imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)

                # Compute moments
                M = cv2.moments(imgMorph)
                # TODO: 1. if computing values not needed try other functions
                #       2. check if CUDA is used (faster), if not use cuda optimized cv functions (or process instead of thread)
                #       3. use a threshold for valid detection so that when no animal detected returns pos2D=[-1,-1]

                if M['m00'] != 0:   # TODO: use small criteria (before it was rounded to an int)
                    # Something detected: use moments for new position (in cropped image)
                    posX = M['m10'] / M['m00']
                    posY = M['m01'] / M['m00']
                else:
                    # Nothing detected (animal moved out of dynamic mask): keep previous position
                    # posX = posXslide[0]
                    # posY = posYslide[0]
                    # Nothing detected (animal moved out of dynamic mask): take center of image
                    posX = self.cropSize[0]//2
                    posY = self.cropSize[1]//2

                # 2D filtering: average over sliding window to stabilize
                if self.filter2D == 1:
                    posXslide = [posX] + posXslide[:-1]
                    posYslide = [posY] + posYslide[:-1]
                    posX = np.mean(posXslide)
                    posY = np.mean(posYslide)

                # Convert values to full image coordinates (keep floating precision)
                pos2D[0] = [self.cropULs[camInd][0] + posX, self.cropULs[camInd][1] + posY]
                nBlobs = 1
                self.log.LogText(3, 'RunDetect(%d): image %03d detected position (X,Y)=(%.1f, %.1f)' % (camNb, imgIndex, pos2D[0, 0], pos2D[0, 1]))

            else:
                # Default blob detector runs on imgDiff
                imgBD = imgDiff

                if self.blobDetectMode in ['thresh', 'morph']:
                    # Thresholds the image (binary)
                    _, imgThresh = cv2.threshold(imgDiff, self.camThresh, 255, cv2.THRESH_BINARY)

                    if self.blobDetectMode in ['morph']:
                        # Morphing
                        imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)
                        imgBD = imgMorph
                    else:
                        imgBD = imgThresh

                # Run blob detector on image
                blobList = blobDetector.detect(imgBD)
                nBlobs = min(len(blobList), self.nBlobsMax)

                # Sort detected blobs by size (better not, artefacts occur in second blob usually)
                if sortBlobsBySize and nBlobs > 1:
                    blobList = [blobList[i[0]].pt for i in sorted(enumerate(blobList), key=lambda x: x[1].size)]
                else:
                    blobList = [blobList[bIndex].pt for bIndex in range(nBlobs)]
                self.log.LogText(3, 'RunDetect(%d): blobList=%s (nBlobs=%d)' % (camNb, str(blobList), nBlobs))

                # Detect mirror blobs
                if removeMirrorBlobs:
                    if self.nFish == 1:
                        # Only one fish
                        if nBlobs == 2:
                            for bIndex in range(0, 2):
                                if borderV(blobList, bIndex):
                                    if alignH(blobList, bIndex, 1-bIndex) and dist(blobList, bIndex, 1-bIndex) < maxDist:  # Less than maxDist px between blobs
                                        blobList.pop(closerV(blobList, bIndex, 1-bIndex))
                                        # self.log.LogText(4, 'RunDetect(%d): removed mirror blob closerV' % camNb)
                                        break
                                elif borderH(blobList, bIndex):
                                    if alignV(blobList, bIndex, 1-bIndex) and dist(blobList, bIndex, 1-bIndex) < maxDist:  # Less than maxDist px between blobs
                                        blobList.pop(closerH(blobList, bIndex, 1-bIndex))
                                        # self.log.LogText(4, 'RunDetect(%d): removed mirror blob closerH' % camNb)
                                        break
                    else:
                        # More than one fish
                        badList = []
                        for bIndex in range(0, nBlobs):
                            if borderV(blobList, bIndex):
                                for bInd2 in range(bIndex, nBlobs):
                                    if alignH(blobList, bIndex, bInd2) and dist(blobList, bIndex, bInd2) < maxDist:  # Less than maxDist px between blobs
                                        blobList.pop(closerV(blobList, bIndex, bInd2))
                            elif borderH(blobList, bIndex):
                                for bInd2 in range(bIndex, nBlobs):
                                    if alignV(blobList, bIndex, bInd2) and dist(blobList, bIndex, bInd2) < maxDist:  # Less than maxDist px between blobs
                                        blobList.pop(closerH(blobList, bIndex, bInd2))
                        blobList = [blobList[bIndex] for bIndex in range(nBlobs) and bIndex not in badList]     # Remove from lisrt

                    self.log.LogText(3, 'RunDetect(%d): removed %d mirror blobs, new blobList=%s (nBlobs=%d)' % (camNb, nBlobs-len(blobList), str(blobList), nBlobs))
                    nBlobs = len(blobList)

                # Stores only valid blob
                for blobIndex in range(nBlobs):
                    pos2D[blobIndex] = [self.cropULs[camInd][0] + blobList[blobIndex][0], self.cropULs[camInd][1] + blobList[blobIndex][1]]

                self.log.LogText(3, 'RunDetect(%d): image %03d detected positions (X,Y)=%s' % (camNb, imgIndex, str(pos2D[:nBlobs])))

            # Stores into shared variables
            self.pos2Ds[camInd] = pos2D.copy()
            self.nFishDetect2D[camInd] = nBlobs                        # Pipes nBlobs to other processes

            # Stores monit images
            if self.imgModes[0] == 'diff':
                self.imgMonits[camInd] = imgDiff.copy()
            elif self.imgModes[0] == 'thresh':
                self.imgMonits[camInd] = imgThresh.copy()
            elif self.imgModes[0] == 'morph':
                self.imgMonits[camInd] = imgMorph.copy()
            if self.dualMode:
                if self.imgModes[1] == 'diff':
                    self.imgMonits[camInd+2] = imgDiff.copy()
                elif self.imgModes[1] == 'thresh':
                    self.imgMonits[camInd+2] = imgThresh.copy()
                elif self.imgModes[1] == 'morph':
                    self.imgMonits[camInd+2] = imgMorph.copy()

    # Run DeepLabCut inferences on cropped images (PROCESS)
    def RunDLC(self):
        """Runs DLC inferences in 2 threads, one for each camera (PROCESS)"""

        self.log.LogText(1, 'RunDLC() process running (PID: %d)' % mp.current_process().pid)

        self.DLCrunning.value = False

        # DLC neural network initialization
        if True:
            from dlclive import DLCLive, Processor

            # DLC inferred Markers vars
            keyCyclopInds = [list(self.keyNames).index(keyName) for keyName in self.keyCyclop]

            if self.modelType == 'base':
                import tensorflow as tf
                # Limite l'expansion abusive de la mémoire par Tensor flow :
                #   - entre 0 et 1, fraction de la mémoire de la carte graphique
                #   - si le même réseau pour tous les process, c'est la même mémoire pour tous, sinon pour chaque réseau
                #   - ne fonctionne pas avec TF RT
                GPUoptions = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
                config = tf.compat.v1.ConfigProto(gpu_options=GPUoptions)

                # Choix du réseau (parmis ceux entrainés) :
                # - dossier du réseau
                # - model_type : base (avec GPU), tensorrt (plus performant, avec GPU), tflite (CPU, pas recommandé)
                # - tf_config : configuration définie juste au dessus (limitation mémoire)
                # - processor : Processor() (peut être redéfini pour ajouter du filtrage prédictif par exemple)
                # - display : pour tracer des points (pas encore trouvé comment, pas utilisé)
                dlc_live = DLCLive(self.neuralNetDir, model_type='base', tf_config=config, processor=Processor(), display=False)

            elif self.modelType == 'pytorch':
                dlc_live = DLCLive(self.neuralNetDir, model_type='pytorch', processor=Processor(), display=False, single_animal=True)

            # Initialize weights with black image (0 info)
            dlc_live.init_inference(np.zeros(self.imgCropDim, dtype=np.float32))        # self.cropSize for 1 channel

        # DLC threaded inference
        def DLC_Inference(self, camInd, dlc_live):
            """Runs DLC inferences on camera camInd (THREAD)"""

            camNb = self.camList[camInd]
            self.log.LogText(1, 'DLC_Inference(%d) thread running' % camNb)
            imgIndexPrev = -1

            while True:

                # Quits on stopRequest
                if self.stopRequest.value or self.quit.value:
                    self.log.LogText(1, 'DLC_Inference(%d): stop request received, thread ending' % camNb)
                    self.DLCrunning.value = False
                    return

                # Get img index and skips if same
                imgIndex = self.imgIndexes[camInd]
                if imgIndex == imgIndexPrev:
                    continue

                # Gets current img
                imgDLC = np.copy(self.imgCrops[camInd])

                # Convert to DLC array format
                imgDLC = np.array(imgDLC, 'float32') / 255    # Values between 0 and 1

                # Computes key positions in cropped image: returns an array (nKeys,3)
                inferredPos = dlc_live.get_pose(imgDLC)

                # Adds UL coordinates
                for keyInd in range(self.nKeys):
                    if inferredPos[keyInd, 2] >= self.pThresh:
                        inferredPos[keyInd, 0] += self.cropULs[camInd][0]
                        inferredPos[keyInd, 1] += self.cropULs[camInd][1]
                    else:
                        # Inference below validity threshold
                        inferredPos[keyInd, 0:2] = -np.ones(2)

                # Stores in shared variable for further triangulation and results saving
                self.pos2Ds_DLC[camInd] = np.copy(inferredPos[:, 0:3])      # DEBUG : now returns 5 per keypoint...
                self.imgIndexes_DLC[camInd] = imgIndex

                # print(inferredPos)
                # print(self.pos2Ds_DLC[camInd])

                # Computes cyclop
                keyAvgValidIndexes = [keyInd for keyInd in keyCyclopInds if self.pos2Ds_DLC[camInd][keyInd, 2] >= self.pThreshCyclop]
                if len(keyAvgValidIndexes) >= 1:      # Minimum of 2 keys in the list must be valid
                    self.pos2D_cyclop[camInd] = np.average(self.pos2Ds_DLC[camInd][keyAvgValidIndexes, :], axis=0)
                else:
                    self.pos2D_cyclop[camInd] = -np.ones(3)
                    # self.log.LogText(4, 'DLC_Inference(%d): inference quality on imgIndex=%d is too low to return a valid cyclop (pos2D)' % (camNb, imgIndex))

        # Run 2 threads, one for each camera
        DLC_threads = []
        for camInd in range(self.camCount):
            DLC_threads.append(threading.Thread(target=DLC_Inference, args=(self, camInd, dlc_live)))
            DLC_threads[camInd].start()

        self.DLCrunning.value = True
        self.log.LogText(1, 'RunDLC: all DLC_inference threads have been launched, quitting')
        return

    # Triangulates pairs of 2D detected positions (PROCESS)
    def Triangulation(self):
        """Triangulates across valid 2D positions (PROCESS)"""

        self.log.LogText(1, 'Triangulation() process running (PID: %d)' % mp.current_process().pid)

        # Loads passage matrices virt2real of the 2-camera system
        pMatrix_VirtToReal = np.load(self.settings.calibDir + 'Pmatrix_virt2real.npy')

        # Loads camera projection matrices
        pMatrices = [None] * self.camCount
        for camNb in range(self.camCount):     # Loads all of them, regardless of connected cameras
            pMatrices[camNb] = np.load(self.settings.calibDir + 'Pmatrix_cam%d.npy' % camNb)

        # Triangulate pos2D pairs
        def Triangulate(pos2D_copy):

            # Builds big matrix with 2 cameras
            A = np.zeros((2 * 2, 4))
            for camInd in range(self.camCount):
                camNb = self.camList[camInd]
                P = pMatrices[camNb]
                A[2*camNb] = pos2D_copy[camInd][0] * P[2] - P[0]        # X
                A[2*camNb + 1] = pos2D_copy[camInd][1] * P[2] - P[1]    # Y

            try:
                # Calls linalg (see book chapter)
                u, d, vt = np.linalg.svd(A)

                # Computes pos3D in cameras virtual reference frame (3*1 matrix)
                pos3Dvirt = vt[-1, 0:3] / vt[-1, 3]

                # Passes pos3D from virtual to real reference frame
                newPos3D = np.dot(pMatrix_VirtToReal[:, :3], pos3Dvirt)
                newPos3D = np.reshape(newPos3D, (3, 1))
                newPos3D += pMatrix_VirtToReal[:, 3:4]  # Why this? must be in the book...
                return newPos3D.T[0]
            except np.linalg.LinAlgError as err:
                self.log.LogText(3, 'Triangulation: LinAlgError: %s' % err)
                return -np.ones(3)

        def OutOfTank(pos3D_copy):
            """Check if a triangulated pos3D is out of the tank (with tolerance)"""

            if self.excludeOutOfTank:
                return np.abs(pos3D_copy[0]) > self.XYmax or np.abs(pos3D_copy[1]) > self.XYmax or pos3D_copy[2] > self.Zmax or pos3D_copy[2] < self.Zmin  # In cm
            else:
                return False

        # Prepare detected 2D possible pair combinations for triangulation (with recursive function)
        if not pairLists:           # (only once)
            def BuildPairList(indList, depth, pairListTmp):
                global pairLists
                b0 = depth
                for b1 in indList:
                    pairListTmp = pairListTmp[:depth]
                    pairListTmp += [(b0, b1)]
                    if depth + 1 == self.nBlobsMax:
                        pairLists = pairLists + [pairListTmp]
                        return
                    else:
                        BuildPairList([b for b in indList if b != b1], depth + 1, pairListTmp)
            BuildPairList(range(self.nBlobsMax), 0, [])

        # Starts connection with Rendering PC
        UDPclientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Initializes sliding window of filter3D
        if self.filter3D != 0:
            pos3Dslide = np.zeros((3, self.filter3Dsize))
        imgIndexesPrev = [-1] * self.camCount
        imgIndexesPrev_DLC = [-1] * self.camCount

        updatePosUDP = False
        while True:

            if self.stopRequest.value:
                self.log.LogText(1, 'Triangulation: stop request received, process ending')
                UDPclientSocket.close()
                return

            # Triangulates detected 2D positions
            if self.runDetect.value:

                # Check if one of the indexes of images has changed and if indexes of images are the same
                if imgIndexesPrev != self.imgIndexes[:]:     # and self.imgIndexes[0] == self.imgIndexes[1]:
                    imgIndexesPrev = self.imgIndexes[:]

                    # Gets current 2D position list and image index
                    pos2Ds_copy = list(self.pos2Ds).copy()
                    nFishDetect2D_copy = list(self.nFishDetect2D).copy()
                    imgIndex0 = self.imgIndexes[0]

                    # Temporary triangulated var
                    pos3Ds_tmp = -np.ones((10, 3))

                    if self.nBlobsMax > 1:
                        # Go through all possible list of combinations
                        pairListStored = []
                        pos3DlistStored = []
                        nFishDetect3Dstored = []

                        self.log.LogText(3, 'Triangulation (imgIndexes=%s): nFishDetect[0]=%d, nFishDetect[1]=%d' % (str(self.imgIndexes), *nFishDetect2D_copy))
                        for pairList in pairLists:

                            # Remove pairs with undetected pos2D set to (0, 0)
                            cleanPairList = pairList.copy()
                            for pair in pairList:
                                # Checks if this pair has valid detection on both cameras
                                if pair[0] >= nFishDetect2D_copy[0] or pair[1] >= nFishDetect2D_copy[1]:
                                    cleanPairList.remove(pair)
                            if not cleanPairList:       # If list is empty, skip
                                break

                            # self.log.LogText(4, 'Triangulation: Testing pairList=%s' % str(cleanPairList))

                            # Triangulate each pair and breaks if out of range
                            pos3Dlist = np.zeros((self.nBlobsMax, 3))        # Stores candidate pos3D list of all triangulated points
                            validPairCounter = 0
                            for pair in cleanPairList:

                                pos2Dpair = np.vstack((pos2Ds_copy[0][pair[0], :], pos2Ds_copy[1][pair[1], :]))     # Get pair
                                pos3Dpair = Triangulate(pos2Dpair)                      # Triangulates
                                if OutOfTank(pos3Dpair):                                # Invalid triangulation, skips this pairlist
                                    # self.log.LogText(4, 'Triangulation: pair (%d, %d) triangulation out of tank' % tuple(pair))
                                    break
                                else:
                                    # self.log.LogText(4, 'Triangulation: pair (%d, %d) triangulation is ok' % tuple(pair))
                                    pos3Dlist[validPairCounter] = pos3Dpair.copy()
                                    validPairCounter += 1

                            if validPairCounter == len(cleanPairList):
                                # self.log.LogText(4, 'Triangulation: pairList=%s is valid' % cleanPairList)
                                pairListStored.append(cleanPairList.copy())
                                pos3DlistStored.append(pos3Dlist.copy())
                                nFishDetect3Dstored.append(validPairCounter)
                            # else:
                            #     self.log.LogText(4, 'Triangulation: pairList=%s is not valid' % cleanPairList)

                        # Process valid pair lists stored
                        nValidPairLists = len(pairListStored)
                        if nValidPairLists == 0:          # Could not find a valid combination list
                            self.log.LogText(3, 'Triangulation (imgIndexes=%s): could not find a pairList with all points remaining in the tank' % str(self.imgIndexes))
                            self.nFishDetect3D.value = 0
                            # pos3D[0] = np.zeros((self.nBlobsMax, 3))              # Take a list of (0,0,0)
                            # pos3D[0] = np.array([[0, 0, 7.5]]*self.nBlobsMax)     # Take a list of small tank center (0,0,7.5)
                        else:
                            # Keeps only first valid pairlist
                            self.nFishDetect3D.value = nFishDetect3Dstored[0]
                            pos3Ds_tmp[:nFishDetect3Dstored[0]] = pos3DlistStored[0][:nFishDetect3Dstored[0]]
                            if not self.useCyclop.value:
                                posUDP = pos3Ds_tmp[0]
                                updatePosUDP = True
                            # if nValidPairLists > 1:
                            #     self.log.LogText(4, 'Triangulation (imgIndexes=%s): found %d valid pairLists, keeping first' % (str(imgIndexes), nValidPairLists))

                    else:
                        # Only one point per camera, triangulate single pair
                        self.log.LogText(3, 'Triangulation (imgIndexes=%s): single pair of detected 2D positions' % str(self.imgIndexes))
                        if self.nFishDetect2D[0] != 0 and self.nFishDetect2D[1] != 0:
                            pos2Dpair = np.vstack((pos2Ds_copy[0][0, :], pos2Ds_copy[1][0, :]))       # Get pair
                            pos3Dpair = Triangulate(pos2Dpair)                              # Triangulates
                            if OutOfTank(pos3Dpair):
                                self.nFishDetect3D.value = 0
                            else:
                                self.nFishDetect3D.value = 1
                                pos3Ds_tmp[0] = pos3Dpair                                   # Valid key triangulation, stores in variable
                                if not self.useCyclop.value:
                                    posUDP = pos3Ds_tmp[0]
                                    updatePosUDP = True
                        else:                                                               # One point is missing for triangulation
                            self.nFishDetect3D.value = 0

                    # self.log.LogText(1, 'Triangulation pos3Ds_tmp=%s' % str(pos3Ds_tmp))
                    self.pos3Ds[0] = pos3Ds_tmp
                    self.imgIndexPos3D.value = imgIndex0            # Takes the index of first camera

            # Triangulates DLC inferred keys
            if self.runDLC.value:

                # Check if one of the indexes of images has changed and if indexes of images are the same
                if imgIndexesPrev_DLC != self.imgIndexes_DLC[:] and self.imgIndexes_DLC[0] == self.imgIndexes_DLC[1]:
                    imgIndexesPrev_DLC = self.imgIndexes_DLC[:]

                    self.log.LogText(3, 'Triangulation (imgIndexes=%s): processing DLC keys' % str(self.imgIndexes))

                    # Gets current 2D DLC position list and image index
                    try:
                        pos2Ds_DLC_copy = np.copy(self.pos2Ds_DLC)                   # Get copy of inferred positions of all cameras
                    except ValueError as err:
                        print('self.pos2Ds_DLC=%s' % self.pos2Ds_DLC)

                    imgIndex0_DLC = self.imgIndexes_DLC[0]

                    # Temporary keys triangulated var
                    pos3Ds_DLC_tmp = np.zeros((self.nKeys, 4))

                    # Triangulates keys
                    for keyInd in range(self.nKeys):
                        pos3Ds_DLC_tmp[keyInd, :3] = Triangulate(pos2Ds_DLC_copy[:2, keyInd, :2])
                        if OutOfTank(pos3Ds_DLC_tmp[keyInd, :3]):
                            pos3Ds_DLC_tmp[keyInd] = -np.ones(4)         # Invalid, sets to -1
                        else:
                            pos3Ds_DLC_tmp[keyInd, 3] = np.mean(pos2Ds_DLC_copy[:2, keyInd, 2])   # Mean inference probability
                    self.pos3Ds_DLC[0] = pos3Ds_DLC_tmp                      # Stores output in manager
                    self.imgIndexPos3D_DLC.value = imgIndex0_DLC             # Takes first camera index

                    self.log.LogText(3, 'Triangulation (imgIndexes=%s): processing cyclop' % str(self.imgIndexes))

                    # Gets current 2D cyclop position list
                    try:
                        pos2D_cyclop_copy = np.copy(self.pos2D_cyclop)           # Get copy of cyclop positions of all cameras
                    except ValueError as err:
                        print('self.pos2D_cyclop=%s' % self.pos2D_cyclop)

                    # Temporary cyclop triangulated var
                    pos3D_cyclop_tmp = np.zeros(4)

                    # Triangulates cyclop
                    if np.all(pos2D_cyclop_copy[:2, :2] != -1):
                        pos3D_cyclop_tmp[:3] = Triangulate(pos2D_cyclop_copy[:2, :2])
                        if OutOfTank(pos3D_cyclop_tmp[:3]):
                            pos3D_cyclop_tmp = -np.ones(4)              # Invalid, sets to -1
                        else:
                            pos3D_cyclop_tmp[3] = np.mean(pos2D_cyclop_copy[:2, 2])     # Mean inference probability
                            if self.useCyclop.value:
                                posUDP = pos3D_cyclop_tmp[:3]
                                updatePosUDP = True
                    else:
                        pos3D_cyclop_tmp = -np.ones(4)                  # One point is missing for triangulation
                    self.pos3D_cyclop[0] = pos3D_cyclop_tmp

            # Send 3D position to rendering via UDP (only valid when updatePosUDP is True)
            if self.sendPos3D.value and updatePosUDP:

                # Filter on pos 3D to be sent : mean or median over sliding window
                if self.filter3D == 1:      # Mean over sliding window
                    pos3Dslide = np.concatenate((pos3Dslide[:, 1:], posUDP.reshape(3, 1)), 1)
                    posUDP = np.mean(pos3Dslide, 1)
                elif self.filter3D == 2:      # Median over sliding window
                    pos3Dslide = np.concatenate((pos3Dslide[:, 1:], posUDP.reshape(3, 1)), 1)
                    posUDP = np.median(pos3Dslide, 1)

                # Send data
                msgOut = '%.3f,%.3f,%.3f' % tuple(posUDP)
                UDPclientSocket.sendto(msgOut.encode(), UDPclientRendering)
                self.log.LogText(3, 'Triangulation: msgOut sent to Rendering \'%s\' (imgIndexe0=%d)' % (msgOut, self.imgIndexes[0]))

                updatePosUDP = False


    # UDP server to receive event's positions and orientations from rendering
    def UDPserver(self):

        self.log.LogText(1, 'UDPserver() thread running (waiting for Rendering event positions/orientations)')

        UDPserverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        UDPserverSocket.bind(('', UDPserverRenderingPort))
        UDPserverSocket.setblocking(False)      # Alternative : UDPserverSocket.settimeout(0)

        # Empty buffer
        try:
            while len(UDPserverSocket.recv(UDPpacketSize)) == 0: pass
        except socket.error:
            pass

        from ast import literal_eval

        while True:

            # Quits on stopRequest
            if self.stopRequest.value or self.quit.value:
                self.log.LogText(1, 'UDPserver: %s received, thread ending' % 'stop request' if self.stopRequest.value else 'quit')
                return

            try:
                msgIn = UDPserverSocket.recv(UDPpacketSize).decode()
                self.log.LogText(3, 'UDPserver: msgIn received \'%s\'' % msgIn)
                textList = msgIn.split('\t')
                textInd = 0
                nEvents = 0
                while textInd < len(textList):
                    if textList[textInd] == 'Time':
                        textInd += 1
                        self.timeRendering.value = float(textList[textInd])
                        # print('self.timeRendering.value = %f' % self.timeRendering.value)           # DEBUG
                        textInd += 1
                    elif textList[textInd] == 'Tracked':
                        textInd += 1
                        self.posTracked[0] = np.array(literal_eval(textList[textInd]))
                        # print('self.posTracked[0] = %s' % self.posTracked[0])                       # DEBUG
                        textInd += 1
                    elif 'Event' in textList[textInd]:
                        eventInd = int(textList[textInd][-1])-1
                        textInd += 1
                        self.posEvents[eventInd] = np.array(literal_eval(textList[textInd]))
                        # print('self.posEvents[%d] = np.array(%s) = %s' % (eventInd, textList[textInd], self.posEvents[eventInd]))     # DEBUG
                        textInd += 1
                        self.rotEvents[eventInd] = np.array(literal_eval(textList[textInd]))
                        # print('self.rotEvents[%d] = np.array(%s) = %s' % (eventInd, textList[textInd], self.rotEvents[eventInd]))     # DEBUG
                        textInd += 1
                        nEvents += 1
                    else:
                        textInd += 1
                self.nEvents.value = nEvents

            except socket.error:
                pass

    # Save results: detected center pos2D and DLC keys (for all cameras), and pos3D triangulations (PROCESS)
    def SaveResults(self, maxDuration=3600):

        self.log.LogText(1, 'SaveResults process running (PID: %d)' % mp.current_process().pid)

        dataRecordingPrev = False               # Stores previous flag state

        # Prepare new results2D array dt
        dt2D = [('time', 'f8')]
        for camInd in range(self.camCount):
            dt2D.extend([('imgIndex_cam%d' % camInd, 'u4'), ('pos(UL)_cam%d' % camInd, 'u2', 2)])
            if self.runDetect.value:
                dt2D.append(('nFishDetected_cam%d' % camInd, 'u1'))
                for fishInd in range(self.nBlobsMax):
                    dt2D.extend([('pos(%d)_cam%d' % (fishInd, camInd), 'f8', 2)])
            if self.runDLC.value:
                dt2D.extend([('pos(Cyclop)_cam%d' % camInd, 'f8', 2), ('proba(Cyclop)_cam%d' % camInd, 'f8')])
                for keyName in self.keyNames:
                    dt2D.extend([('pos(%s)_cam%d' % (keyName, camInd), 'f8', 2), ('proba(%s)_cam%d' % (keyName, camInd), 'f8')])
        dt2D = np.dtype(dt2D)

        # Prepare new results3D array dt
        dt3D = [('time', 'f8'), ('imgIndex', 'u4')]  # for new res3D array
        if self.runDetect.value:
            dt3D.append(('nFishDetected', 'u1'))
            for fishInd in range(self.nBlobsMax):
                dt3D.extend([('pos(%d)' % fishInd, 'f8', 3)])
            if not self.runDLC.value and self.getVelocity:
                dt3D.extend([('vel(Cyclop)', 'f8', 3), ('velNorm(Cyclop)', 'f8')])
        if self.runDLC.value:
            dt3D.extend([('pos(Cyclop)', 'f8', 3), ('proba(Cyclop)', 'f8')])
            if self.getVelocity:
                dt3D.extend([('vel(Cyclop)', 'f8', 3), ('velNorm(Cyclop)', 'f8')])
            if self.getMotionDir:
                dt3D.extend([('motionDir', 'i1')])
            if self.getGazeDir:
                dt3D.extend([('gazeDir', 'f8', 3)])
            if self.getCurvature:
                dt3D.extend([('curvature', 'f8')])
            for keyName in self.keyNames:
                dt3D.extend([('pos(%s)' % keyName, 'f8', 3), ('proba(%s)' % keyName, 'f8')])

            # DLC inferred Markers vars
            if self.getGazeDir:
                keyGazeDirInds = [list(self.keyNames).index(keyName) for keyName in self.keyGazeDir]
            if self.getCurvature:
                keyCurvatureInds = [list(self.keyNames).index(keyName) for keyName in self.keyCurvature]

        if self.recvEventPos.value:
            dt3D.extend([('timeRendering', 'f8')])
            dt3D.extend([('pos(tracked)', 'f8', 3)])
            for eventInd in range(nEventsMax):
                dt3D.extend([('pos(event%d)' % eventInd, 'f8', 3)])
                dt3D.extend([('rot(event%d)' % eventInd, 'f8', 3)])

        dt3D = np.dtype(dt3D)

        while True:

            # Quits on stopRequest
            if self.stopRequest.value:
                self.dataRecording.value = False     # Sends signal to stop data recording (in case)
                if not dataRecordingPrev:       # Quits when no longer recording
                    self.log.LogText(1, 'SaveResults: stop request received, process ending')
                    return

            if self.dataRecording.value:

                if not dataRecordingPrev:
                    # At trial start: initializes arrays
                    self.log.LogText(2, 'SaveResults: preparing arrays (flag dataRecording=True)')

                    # Creates paths if not existing
                    path = self.settings.resultsDir + '%s/%s/' % (self.expID.value, self.subjectID.value)
                    resultsFile = 'Trial-%s_Cond-%s' % (self.trialID.value, self.condID.value)
                    try:
                        os.makedirs(path)
                    except FileExistsError:
                        pass

                    # Prepare arrays (limited to maxDuration of recording)
                    res2D = np.recarray((int(self.framerate * maxDuration),), dtype=dt2D)
                    res2D['time'] = -1.0
                    res3D = np.recarray((int(self.framerate * maxDuration,)), dtype=dt3D)
                    res3D['time'] = -1.0

                    # Store initial index of cam0 for time column and rowIndex computations
                    indexStart = 0      # was self.imgIndexes[0] or pos2DimgIndexes[0]
                    imgIndexesPrev = [-1] * self.camCount
                    imgIndexesPrev_DLC = [-1] * self.camCount
                    imgIndexPos3DPrev = -1
                    imgIndexPos3DPrev_DLC = -1

                    dataRecordingPrev = True

                # During trial: records data

                # 2D data
                for camInd in range(self.camCount):
                    if self.runDetect.value:
                        if self.imgIndexes[camInd] != imgIndexesPrev[camInd]:
                            imgIndexesPrev[camInd] = self.imgIndexes[camInd]
                            fInd = int(self.imgIndexes[camInd] - indexStart)
                            res2D['time'][fInd] = fInd / self.framerate
                            res2D['imgIndex_cam%d' % camInd][fInd] = fInd
                            res2D['pos(UL)_cam%d' % camInd][fInd] = self.cropULs[camInd]
                            res2D['nFishDetected_cam%d' % camInd][fInd] = self.nFishDetect2D[camInd]
                            for fishInd in range(self.nFishDetect2D[camInd]):                                # Undetected remain at 0
                                res2D['pos(%d)_cam%d' % (fishInd, camInd)][fInd] = self.pos2Ds[camInd][fishInd]               # X, Y (pos2D)
                            self.log.LogText(3, 'SaveResults: new data available, updating results arrays ')
                    if self.runDLC.value:
                        if self.imgIndexes_DLC[camInd] != imgIndexesPrev_DLC[camInd]:
                            imgIndexesPrev_DLC[camInd] = self.imgIndexes_DLC[camInd]
                            fInd = int(self.imgIndexes_DLC[camInd] - indexStart)
                            res2D['time'][fInd] = fInd / self.framerate
                            res2D['imgIndex_cam%d' % camInd][fInd] = fInd
                            res2D['pos(UL)_cam%d' % camInd][fInd] = self.cropULs[camInd]
                            res2D['pos(Cyclop)_cam%d' % camInd][fInd] = self.pos2D_cyclop[camInd][:2]  # X, Y (pos2D) of cyclop
                            res2D['proba(Cyclop)_cam%d' % camInd][fInd] = self.pos2D_cyclop[camInd][2]  # pMean of cyclop
                            for keyInd, keyName in enumerate(self.keyNames):
                                res2D['pos(%s)_cam%d' % (keyName, camInd)][fInd] = self.pos2Ds_DLC[camInd][keyInd, :2]      # X, Y (pos2D) of key keyInd
                                res2D['proba(%s)_cam%d' % (keyName, camInd)][fInd] = self.pos2Ds_DLC[camInd][keyInd, 2]     # pMean of key keyInd

                # 3D data
                if self.triangulate.value:
                    if self.runDetect.value:
                        if self.imgIndexPos3D.value != imgIndexPos3DPrev:
                            imgIndexPos3DPrev = self.imgIndexPos3D.value
                            fInd = int(self.imgIndexPos3D.value - indexStart)
                            res3D['time'][fInd] = fInd / self.framerate
                            res3D['imgIndex'][fInd] = fInd
                            res3D['nFishDetected'][fInd] = self.nFishDetect3D.value
                            for fishInd in range(self.nFishDetect3D.value):          # Undetected remain at 0
                                res3D['pos(%d)' % fishInd][fInd] = self.pos3Ds[0][fishInd]       # Stores in results array X, Y, Z (pos3D)
                            if not self.runDLC.value and self.getVelocity and fInd > 0:
                                # Instant velocity vector and norm
                                if np.all(res3D['pos(0)'][fInd] != -1) and np.all(res3D['pos(0)'][fInd-1]) != -1 and np.all(res3D['pos(0)'][fInd-1] != 0):
                                    res3D['vel(0)'][fInd] = (res3D['pos(0)'][fInd] - res3D['pos(0)'][fInd-1]) / (res3D['time'][fInd] - res3D['time'][fInd-1])
                                    res3D['velNorm(0)'][fInd] = np.linalg.norm(res3D['vel(0)'][fInd])
                                else:
                                    res3D['vel(0)'][fInd] = -1
                                    res3D['velNorm(0)'][fInd] = -1
                            # Feedback from rendering computer (if runDLC is false)
                            if self.recvEventPos.value and not self.runDLC.value:
                                res3D['timeRendering'][fInd] = self.timeRendering.value
                                res3D['pos(tracked)'][fInd] = self.posTracked[0]
                                for eventInd in range(self.nEvents.value):
                                    res3D['pos(event%d)' % eventInd][fInd] = self.posEvents[eventInd]
                                    res3D['rot(event%d)' % eventInd][fInd] = self.rotEvents[eventInd]
                                for eventInd in range(self.nEvents.value, nEventsMax):
                                    res3D['pos(event%d)' % eventInd][fInd] = np.zeros(3)
                                    res3D['rot(event%d)' % eventInd][fInd] = np.zeros(3)
                    if self.runDLC.value:
                        if self.imgIndexPos3D_DLC.value != imgIndexPos3DPrev_DLC:
                            imgIndexPos3DPrev_DLC = self.imgIndexPos3D_DLC.value
                            fInd = int(self.imgIndexPos3D_DLC.value - indexStart)
                            res3D['time'][fInd] = fInd / self.framerate
                            res3D['imgIndex'][fInd] = fInd
                            res3D['pos(Cyclop)'][fInd] = self.pos3D_cyclop[0][:3]  # X, Y, Z (pos3D) of cyclop
                            res3D['proba(Cyclop)'][fInd] = self.pos3D_cyclop[0][3]  # pMean of cyclop
                            for keyInd, keyName in enumerate(self.keyNames):
                                res3D['pos(%s)' % keyName][fInd] = self.pos3Ds_DLC[0][keyInd, :3]      # X, Y, Z (pos3D) of key keyInd
                                res3D['proba(%s)' % keyName][fInd] = self.pos3Ds_DLC[0][keyInd, 3]     # pMean of key keyInd
                            if self.getVelocity and fInd > 0:
                                # Instant velocity vector and norm
                                if np.all(res3D['pos(Cyclop)'][fInd] != -1) and np.all(res3D['pos(Cyclop)'][fInd-1]) != -1 and np.all(res3D['pos(Cyclop)'][fInd-1] != 0):
                                    res3D['vel(Cyclop)'][fInd] = (res3D['pos(Cyclop)'][fInd] - res3D['pos(Cyclop)'][fInd-1]) / (res3D['time'][fInd] - res3D['time'][fInd-1])
                                    res3D['velNorm(Cyclop)'][fInd] = np.linalg.norm(res3D['vel(Cyclop)'][fInd])
                                else:
                                    res3D['vel(Cyclop)'][fInd] = -1
                                    res3D['velNorm(Cyclop)'][fInd] = -1
                            if self.getGazeDir:
                                # Instant gaze direction (normalized vector)
                                vStart, vEnd = self.pos3Ds_DLC[0][keyGazeDirInds, :3]
                                if np.all(vStart != -1) and np.all(vEnd != -1):
                                    v = vEnd - vStart
                                    res3D['gazeDir'][fInd] = v / np.linalg.norm(v)
                                else:
                                    res3D['gazeDir'][fInd] = -1
                            if self.getMotionDir:
                                # Direction of motion: forward=+1, backward=-1, static=0
                                if np.all(res3D['vel(Cyclop)'][fInd] != -1) and np.all(res3D['gazeDir'][fInd] != -1):
                                    if np.dot(res3D['vel(Cyclop)'][fInd], res3D['gazeDir'][fInd]) > 0:
                                        res3D['motionDir'][fInd] = +1 if res3D['velNorm(Cyclop)'][fInd] > self.minFwdSpeed else 0
                                    else:
                                        res3D['motionDir'][fInd] = -1 if res3D['velNorm(Cyclop)'][fInd] > self.minBwdSpeed else 0
                                else:
                                    res3D['motionDir'][fInd] = 0
                            if self.getCurvature:
                                vTailStart, vTailEnd, vHeadStart, vHeadEnd = self.pos3Ds_DLC[0][keyCurvatureInds, :3]
                                if np.all(vTailStart != -1) and np.all(vTailEnd != -1) and np.all(vHeadStart != -1) and np.all(vHeadEnd != -1):
                                    vTail = vTailEnd - vTailStart
                                    vHead = vHeadEnd - vHeadStart
                                    # !!! There is no obvious way to get the sign for motion in 3D !!!
                                    res3D['curvature'][fInd] = np.arccos(np.dot(vTail, vHead) / np.linalg.norm(vTail) / np.linalg.norm(vHead)) * 180.0 / np.pi
                                    # res3D['curvature'][fInd] = np.arcsin(np.linalg.norm(np.cross(vTail, vHead)) / np.linalg.norm(vTail) / np.linalg.norm(vHead)) * 180.0 / np.pi
                                else:
                                    res3D['curvature'][fInd] = -1
                            # Feedback from rendering computer (if runDLC is false)
                            if self.recvEventPos.value:
                                res3D['timeRendering'][fInd] = self.timeRendering.value
                                res3D['pos(tracked)'][fInd] = self.posTracked[0]
                                for eventInd in range(self.nEvents.value):
                                    res3D['pos(event%d)' % eventInd][fInd] = self.posEvents[eventInd]
                                    res3D['rot(event%d)' % eventInd][fInd] = self.rotEvents[eventInd]
                                for eventInd in range(self.nEvents.value, nEventsMax):
                                    res3D['pos(event%d)' % eventInd][fInd] = np.zeros(3)
                                    res3D['rot(event%d)' % eventInd][fInd] = np.zeros(3)

            elif dataRecordingPrev:
                # At trial end: stores arrays (append writing mode)
                self.log.LogText(2, 'SaveResults: saving arrays (flag dataRecording=False)')

                # General results log file
                header = ['expID', 'subjectID', 'trialID', 'condID', 'filename']
                fileName = self.settings.resultsDir + '%s/' % self.expID.value + '%s_files.tsv' % self.expID.value
                writeHeader = not os.path.isfile(fileName)
                with open(fileName, 'a') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter='\t')
                    if writeHeader:
                        filewriter.writerow(header)
                    filewriter.writerow([self.expID.value, self.subjectID.value, self.trialID.value, self.condID.value, resultsFile])

                # 2D data
                if self.runDetect.value:
                    if self.runDLC.value:
                        filename = path + resultsFile + '_pos2D+DLC2D'
                    else:
                        filename = path + resultsFile + '_pos2D'
                else:
                    filename = path + resultsFile + '_DLC2D'
                res2D = res2D[res2D['time'] != -1.0]
                np.save(filename, res2D)
                if self.settings.saveTextCopy:
                    header2D = ('%s\t' * len(dt2D.names)) % dt2D.names
                    np.savetxt(filename + '.tsv', res2D, fmt='%s', delimiter='\t', header=header2D, comments='')
                    # res2D.tofile(filename + '.txt', sep='\n')

                # 3D data
                if self.triangulate.value:
                    if self.runDetect.value:
                        if self.runDLC.value:
                            filename = path + resultsFile + '_pos3D+DLC3D'
                        else:
                            filename = path + resultsFile + '_pos3D'
                    else:
                        filename = path + resultsFile + '_DLC3D'
                    res3D = res3D[res3D['time'] != -1.0]
                    np.save(filename, res3D)
                    if self.settings.saveTextCopy:
                        header3D = ('%s\t' * len(dt3D.names)) % dt3D.names
                        np.savetxt(filename + '.tsv', res3D, fmt='%s', delimiter='\t', header=header3D, comments='')
                        # res3D.tofile(filename + '.txt', sep='\n')

                dataRecordingPrev = False
                self.dataRecording.value = False

    # Launches the Graphic User Interface (PROCESS)
    def StartGUI(self):

        self.log.LogText(1, 'StartGUI: creating UIController object')
        myQtApp = QApplication(sys.argv)
        UI = UIController(self_tracking=self)
        myQtApp.exec_()


# GUI for tracking
class UIController(QWidget):

    def __init__(self, self_tracking):
        """Constructor"""

        # Get log object
        self.log = self_tracking.log
        self.log.LogText(1, 'UIController() called')

        # UI shared controller vars
        UImode = self_tracking.settings.controller == 'UI'
        if UImode:
            self.expID = self_tracking.expID
            self.subjectID = self_tracking.subjectID
            self.trialID = self_tracking.trialID
            self.condID = self_tracking.condID
        self.speciesName = self_tracking.speciesName
        self.runDetect = self_tracking.runDetect
        self.runDLC = self_tracking.runDLC
        self.triangulate = self_tracking.triangulate
        self.showPos2D = self_tracking.showPos2D
        self.showDLC = self_tracking.showDLC
        self.useCyclop = self_tracking.useCyclop
        self.sendPos3D = self_tracking.sendPos3D
        self.recvEventPos = self_tracking.recvEventPos
        self.saveResults = self_tracking.saveResults
        self.imgModes = self_tracking.imgModes
        self.stopRequest = self_tracking.stopRequest
        self.quit = self_tracking.quit

        self.settings = self_tracking.settings
        self.self_tracking = self_tracking

        super().__init__()

        # Load section UIs
        prevY = self.SystemSettingsUI(posX=10, posY=0, UImode=UImode)
        if UImode:
            prevY = self.ExperimentSettingsUI(posX=10, posY=prevY+10)
            self.ControllerUI(posX=10, posY=prevY+10)

        # Camera return visual
        self.panel = QLabel(self)

        # General aspect of the window
        self.setFixedSize(260, 710 if UImode else 440)
        self.move(10, 10)
        self.setWindowTitle('Tracking UI')
        self.show()

    def SystemSettingsUI(self, posX, posY, UImode=False):
        # Section title
        self.systemSetLbl = QLabel('System settings', self)
        self.systemSetLbl.setGeometry(posX, posY, 150, 30)
        self.systemSetLbl.setStyleSheet("font-weight: bold")
        posY += 40
        # Fish species name
        self.speciesNameTxt = QLabel('Species', self)
        self.speciesNameTxt.setGeometry(posX+20, posY, 120, 30)
        self.speciesNameCombo = QComboBox(self)
        self.speciesNameCombo.addItems(self.settings.speciesNameList)
        self.speciesNameCombo.setGeometry(posX + 70, posY, 150, 30)
        self.speciesNameCombo.setCurrentIndex(self.settings.speciesNameList.index(self.speciesName.value))
        self.speciesNameCombo.currentIndexChanged.connect(self.SpeciesNames)
        posY += 45
        # Run detection
        self.runDetectBtn = QPushButton('Simple detect', self)
        self.runDetectBtn.setGeometry(posX, posY, 120, 30)
        self.runDetectBtn.setCheckable(True)
        self.runDetectBtn.setChecked(self.runDetect.value)
        self.runDetectBtn.clicked.connect(self.RunDetect)
        # Show detected position
        self.showPos2DBtn = QPushButton('Show 2D pos', self)
        self.showPos2DBtn.setGeometry(posX + 120, posY, 120, 30)
        self.showPos2DBtn.setCheckable(True)
        self.showPos2DBtn.setChecked(self.showPos2D.value)
        self.showPos2DBtn.setEnabled(self.runDetect.value or self.runDLC.value)
        self.showPos2DBtn.clicked.connect(self.ShowPos2D)
        posY += 35
        # Run DeepLabCut
        self.runDLCBtn = QPushButton('DeepLabCut', self)
        self.runDLCBtn.setGeometry(posX, posY, 120, 30)
        self.runDLCBtn.setCheckable(True)
        self.runDLCBtn.setChecked(self.runDLC.value)
        self.runDLCBtn.clicked.connect(self.RunDLC)
        # Show DeepLabCut
        self.showDLCBtn = QPushButton('Show keys', self)
        self.showDLCBtn.setGeometry(posX + 120, posY, 120, 30)
        self.showDLCBtn.setCheckable(True)
        self.showDLCBtn.setEnabled(self.runDLC.value)
        self.showDLCBtn.setChecked(self.showDLC.value)
        self.showDLCBtn.clicked.connect(self.ShowDLC)
        posY += 35
        # Use cyclop checker
        self.useCyclopChk = QCheckBox('Use cyclop', self)
        self.useCyclopChk.setCheckable(True)
        self.useCyclopChk.setChecked(self.useCyclop.value)
        self.useCyclopChk.setEnabled(self.runDLC.value)
        self.useCyclopChk.setGeometry(posX + 100, posY, 180, 30)
        self.useCyclopChk.clicked.connect(self.UseCyclop)
        posY += 45
        # Run triangulation
        self.triangulateBtn = QPushButton('Triangulate', self)
        self.triangulateBtn.setGeometry(posX, posY, 80, 30)
        self.triangulateBtn.setCheckable(True)
        self.triangulateBtn.setChecked(self.triangulate.value)
        self.triangulateBtn.setEnabled(self.runDetect.value or self.runDLC.value)
        self.triangulateBtn.clicked.connect(self.Triangulate)
        # Send 3D position to Unreal
        self.sendPos3DBtn = QPushButton('Send fish', self)
        self.sendPos3DBtn.setGeometry(posX + 80, posY, 80, 30)
        self.sendPos3DBtn.setCheckable(True)
        self.sendPos3DBtn.setEnabled(self.triangulate.value)
        self.sendPos3DBtn.setChecked(self.sendPos3D.value)
        self.sendPos3DBtn.clicked.connect(self.SendPos3D)
        # Receive stimulus 3D pos/rot from Unreal
        self.recvEventPos3DBtn = QPushButton('Receive events', self)
        self.recvEventPos3DBtn.setGeometry(posX + 160, posY, 80, 30)
        self.recvEventPos3DBtn.setCheckable(True)
        # self.recvEventPos3DBtn.setEnabled(False)        # To block feature
        self.recvEventPos3DBtn.setEnabled(self.sendPos3D.value)
        self.recvEventPos3DBtn.setChecked(self.recvEventPos.value)
        self.recvEventPos3DBtn.clicked.connect(self.RecvEventPos)
        posY += 40
        # Image Mode selection
        self.imgModeTxt0 = QLabel('Upper monitoring', self)
        self.imgModeTxt0.setGeometry(posX+20, posY, 120, 30)
        self.imgModeCombo0 = QComboBox(self)
        # self.imgType0 = ['crop', 'diff', 'thresh', 'morph']
        self.imgType0 = ['full', 'crop', 'diff', 'thresh', 'morph']
        self.imgModeCombo0.addItems(self.imgType0)
        self.imgModeCombo0.setGeometry(posX + 150, posY, 90, 30)
        self.imgModeCombo0.setCurrentIndex(self.imgType0.index(self.imgModes[0]))
        self.imgModeCombo0.currentIndexChanged.connect(self.ImgModes)
        posY += 35
        # Image Mode selection
        self.imgModeTxt1 = QLabel('Lower monitoring', self)
        self.imgModeTxt1.setGeometry(posX+20, posY, 120, 30)
        self.imgModeCombo1 = QComboBox(self)
        self.imgType1 = ['none', 'crop', 'diff', 'thresh', 'morph']
        self.imgModeCombo1.addItems(self.imgType1)
        self.imgModeCombo1.setGeometry(posX + 150, posY, 90, 30)
        if not self.runDetect.value or self.imgModes[0] == 'full':
            self.imgModeTxt1.setEnabled(False)
            self.imgModeCombo1.setEnabled(False)
            self.imgModes[1] = 'none'
        self.imgModeCombo1.setCurrentIndex(self.imgType1.index(self.imgModes[1]))
        self.imgModeCombo1.currentIndexChanged.connect(self.ImgModes)
        posY += 45
        # Load settings
        self.loadSettingsBtn = QPushButton('Reload Settings', self)
        self.loadSettingsBtn.setGeometry(posX + 50, posY, 140, 30)
        self.loadSettingsBtn.setEnabled(True)
        self.loadSettingsBtn.clicked.connect(self.LoadSettings)
        posY += 35

        # In UE controller mode, add...
        if not UImode:
            # New reference image button
            posY += 10
            self.newRefBtn = QPushButton('New Reference Image', self)
            self.newRefBtn.setGeometry(posX+30, posY, 180, 30)
            self.newRefBtn.setEnabled(False)
            self.newRefBtn.clicked.connect(self.NewRef)

            posY += 40
            # Save results checker
            self.saveResultsChk = QCheckBox('Save results', self)
            self.saveResultsChk.setCheckable(True)
            self.saveResultsChk.setChecked(self.saveResults.value)
            self.saveResultsChk.setEnabled(True)
            self.saveResultsChk.setGeometry(posX + 100, posY, 180, 30)
            self.saveResultsChk.clicked.connect(self.SaveResults)
            posY += 30

        return posY

    def UpdateSystemSettingsUI(self):

        self.runDetectBtn.setChecked(self.runDetect.value)
        self.showPos2DBtn.setChecked(self.showPos2D.value)
        self.triangulateBtn.setChecked(self.triangulate.value)
        if self.triangulate.value:
            self.sendPos3DBtn.setEnabled(True)
            self.sendPos3DBtn.setChecked(self.sendPos3D.value)
        self.runDLCBtn.setChecked(self.runDLC.value)
        if self.runDLC.value:
            self.showDLCBtn.setEnabled(True)
            self.showDLCBtn.setChecked(self.showDLC.value)
        self.imgModeCombo0.setCurrentIndex(self.imgType0.index(self.imgModes[0]))
        self.imgModeCombo1.setCurrentIndex(self.imgType1.index(self.imgModes[1]))
        self.speciesNameCombo.setCurrentIndex(self.settings.speciesNameList.index(self.speciesName.value))

        self.saveResultsChk.setChecked(self.saveResults.value)

        self.log.LogText(2, 'UpdateSystemSettingsUI: done')

    def ExperimentSettingsUI(self, posX, posY):
        # Section title
        self.experimentSetLbl = QLabel('Experiment Settings', self)
        self.experimentSetLbl.setGeometry(posX, posY, 150, 30)
        self.experimentSetLbl.setStyleSheet("font-weight: bold")
        posY += 40
        # Experiment ID
        self.expID_label = QLabel('Experiment', self)
        self.expID_label.setGeometry(posX, posY, 100, 20)
        self.expID_text = QLineEdit(self)
        self.expID_text.setText(self.expID.value)
        self.expID_text.setGeometry(posX + 90, posY, 150, 20)
        posY += 25
        # Subject ID
        self.subjID_label = QLabel('Subject', self)
        self.subjID_label.setGeometry(posX, posY, 100, 20)
        self.subjID_text = QLineEdit(self)
        self.subjID_text.setText(self.subjectID.value)
        self.subjID_text.setGeometry(posX + 90, posY, 150, 20)
        posY += 25
        # TrialID
        self.trialID_label = QLabel('Trial', self)
        self.trialID_label.setGeometry(posX, posY, 100, 20)
        self.trialID_text = QLineEdit(self)
        self.trialID_text.setText(self.trialID.value)
        self.trialID_text.setGeometry(posX + 90, posY, 150, 20)
        posY += 25
        # ConditionID
        self.condID_label = QLabel('Condition', self)
        self.condID_label.setGeometry(posX, posY, 100, 20)
        self.condID_text = QLineEdit(self)
        self.condID_text.setText(self.condID.value)
        self.condID_text.setGeometry(posX + 90, posY, 150, 20)
        posY += 30

        return posY

    def UpdateExperimentSettingsUI(self):

        # self.textExpID.setText(self.expID.value)
        # self.textSubjID.setText(self.subjectID.value)
        # self.textTrialID.setText(self.trialID.value)
        # self.textCondID.setText(self.condID.value)
        self.saveResultsChk.setChecked(self.saveResults.value)

        self.log.LogText(2, 'UpdateExperimentSettingsUI: done')

    def ControllerUI(self, posX, posY):
        # Section title
        self.controllerLbl = QLabel('Controller', self)
        self.controllerLbl.setGeometry(posX, posY, 150, 30)
        self.controllerLbl.setStyleSheet("font-weight: bold")
        posY += 30
        # Start experiment
        self.startExpBtn = QPushButton('Start Experiment', self)
        self.startExpBtn.setGeometry(posX, posY, 120, 30)
        self.startExpBtn.setCheckable(True)
        self.startExpBtn.setEnabled(True)
        self.startExpBtn.clicked.connect(self.StartExperiment)
        # End experiment
        self.endExpBtn = QPushButton('End Experiment', self)
        self.endExpBtn.setGeometry(posX + 120, posY, 120, 30)
        self.endExpBtn.setCheckable(True)
        self.endExpBtn.setEnabled(False)
        self.endExpBtn.clicked.connect(self.EndExperiment)
        posY += 35
        # Start trial
        self.startTrialBtn = QPushButton('Start Trial', self)
        self.startTrialBtn.setGeometry(posX, posY, 120, 30)
        self.startTrialBtn.setEnabled(False)
        self.startTrialBtn.clicked.connect(self.StartTrial)
        # End trial
        self.endTrialBtn = QPushButton('End Trial', self)
        self.endTrialBtn.setGeometry(posX + 120, posY, 120, 30)
        self.endTrialBtn.setEnabled(False)
        self.endTrialBtn.clicked.connect(self.EndTrial)
        posY += 45
        # New reference image
        self.newRefBtn = QPushButton('New Reference Image', self)
        self.newRefBtn.setGeometry(posX + 30, posY, 180, 30)
        self.newRefBtn.setEnabled(False)
        self.newRefBtn.clicked.connect(self.NewRef)

        posY += 40
        # Save results checker
        self.saveResultsChk = QCheckBox('Save results', self)
        self.saveResultsChk.setGeometry(posX + 100, posY, 180, 30)
        self.saveResultsChk.setCheckable(True)
        self.saveResultsChk.setChecked(self.saveResults.value)
        self.saveResultsChk.setEnabled(True)
        self.saveResultsChk.clicked.connect(self.SaveResults)
        posY += 30

        return posY

    def SpeciesNames(self):
        self.speciesName.value = self.speciesNameCombo.currentText()

    def ImgModes(self):
        self.imgModes[:] = [self.imgModeCombo0.currentText(), self.imgModeCombo1.currentText()]
        if self.imgModes[0] == 'full':
            self.imgModes[1] = 'none'
            self.imgModeTxt1.setEnabled(False)
            self.imgModeCombo1.setEnabled(False)
            self.imgModeCombo1.setCurrentIndex(self.imgType1.index(self.imgModes[1]))
        else:
            self.imgModeTxt1.setEnabled(True)
            self.imgModeCombo1.setEnabled(True)

    def StartExperiment(self):
        # Disable the monitoring when a trial is started
        self.speciesNameCombo.setEnabled(False)
        self.startExpBtn.setEnabled(False)
        self.startExpBtn.setChecked(False)
        self.endExpBtn.setEnabled(True)
        self.endExpBtn.setCheckable(True)
        self.endExpBtn.setChecked(False)
        self.startTrialBtn.setEnabled(True)
        self.newRefBtn.setEnabled(self.runDetect.value)      # Enable newRef button if runDetect is True
        self.runDLCBtn.setEnabled(False)
        self.runDetectBtn.setEnabled(False)
        self.triangulateBtn.setEnabled(False)
        self.imgModeCombo0.setEnabled(False)
        self.imgModeCombo1.setEnabled(False)
        self.showPos2DBtn.setEnabled(True)
        # Send command to TCP server
        self.SendCommandTCPTracking('startExperiment')
        self.log.LogText(2, 'UIController: startExperiment sent')

    def EndExperiment(self):
        self.speciesNameCombo.setEnabled(True)
        self.startExpBtn.setEnabled(True)
        self.startExpBtn.setChecked(False)
        self.endExpBtn.setEnabled(False)
        self.endExpBtn.setChecked(False)
        self.newRefBtn.setEnabled(False)            # Disable newRef button
        self.runDLCBtn.setEnabled(True)
        self.runDetectBtn.setEnabled(True)
        self.triangulateBtn.setEnabled(True)
        self.startTrialBtn.setChecked(False)
        self.startTrialBtn.setEnabled(False)
        self.endTrialBtn.setEnabled(False)
        self.imgModeCombo0.setEnabled(True)
        self.imgModeCombo1.setEnabled(True)
        # Send command to TCP server
        self.SendCommandTCPTracking('endExperiment')
        self.log.LogText(2, 'UIController: endExperiment sent')

    def StartTrial(self):
        self.startTrialBtn.setChecked(False)
        self.startTrialBtn.setEnabled(False)
        self.endTrialBtn.setEnabled(True)               # Enable endTrial button
        self.saveResultsChk.setEnabled(False)           # Disable saveResults checkbox
        # Disable experiment UI fields
        self.expID_text.setEnabled(False)
        self.subjID_text.setEnabled(False)
        self.trialID_text.setEnabled(False)
        self.condID_text.setEnabled(False)
        # Send commands to TCP server
        self.SendCommandTCPTracking('expID=%s' % self.expID_text.text())
        self.SendCommandTCPTracking('subjectID=%s' % self.subjID_text.text())
        self.SendCommandTCPTracking('condID=%s' % self.condID_text.text())
        self.SendCommandTCPTracking('trialID=%s' % self.trialID_text.text())
        self.SendCommandTCPTracking('startTrial')
        self.log.LogText(2, 'UIController: startTrial sent')

    def EndTrial(self):
        self.endTrialBtn.setEnabled(False)
        self.endTrialBtn.setChecked(False)
        self.startTrialBtn.setEnabled(True)         # Enable startTrial button
        self.saveResultsChk.setEnabled(True)        # Enable saveResults checkbox
        # Enable experiment UI fields
        self.expID_text.setEnabled(True)
        self.subjID_text.setEnabled(True)
        self.trialID_text.setEnabled(True)
        self.condID_text.setEnabled(True)
        # Send command to TCP server
        self.SendCommandTCPTracking('endTrial')
        self.log.LogText(2, 'UIController: endTrial sent')

    def RunDetect(self):
        if self.runDetectBtn.isChecked():
            self.imgModeTxt1.setEnabled(True)
            self.imgModeCombo1.setEnabled(True)
            self.imgModes[1] = 'diff'
            if not self.runDLC.value:
                self.showPos2DBtn.setChecked(True)
                self.showPos2DBtn.setEnabled(True)
                self.triangulateBtn.setChecked(True)
                self.triangulateBtn.setEnabled(True)
                self.sendPos3DBtn.setChecked(False)
                self.sendPos3DBtn.setEnabled(True)
        else:
            self.imgModeTxt1.setEnabled(False)
            self.imgModeCombo1.setEnabled(False)
            self.imgModes[1] = 'none'
            self.newRefBtn.setEnabled(False)
            if not self.runDLC.value:
                self.showPos2DBtn.setChecked(False)
                self.showPos2DBtn.setEnabled(False)
                self.triangulateBtn.setChecked(False)
                self.triangulateBtn.setEnabled(False)
                self.sendPos3DBtn.setChecked(False)
                self.sendPos3DBtn.setEnabled(False)
        self.imgModeCombo1.setCurrentIndex(self.imgType1.index(self.imgModes[1]))
        self.runDetect.value = self.runDetectBtn.isChecked()
        self.showPos2D.value = self.showPos2DBtn.isChecked()
        self.triangulate.value = self.triangulateBtn.isChecked()
        self.sendPos3D.value = self.sendPos3DBtn.isChecked()

    def RunDLC(self):
        if self.runDLCBtn.isChecked():
            self.showDLCBtn.setChecked(True)
            self.showDLCBtn.setEnabled(True)
            self.useCyclopChk.setChecked(True)
            self.useCyclopChk.setEnabled(True)
            if not self.runDetect.value:
                self.showPos2DBtn.setChecked(True)
                self.showPos2DBtn.setEnabled(True)
                self.triangulateBtn.setChecked(True)
                self.triangulateBtn.setEnabled(True)
                self.sendPos3DBtn.setChecked(False)
                self.sendPos3DBtn.setEnabled(True)
        else:
            self.showDLCBtn.setChecked(False)
            self.showDLCBtn.setEnabled(False)
            self.useCyclopChk.setChecked(False)
            self.useCyclopChk.setEnabled(False)
            if not self.runDetect.value:
                self.showPos2DBtn.setChecked(False)
                self.showPos2DBtn.setEnabled(False)
                self.triangulateBtn.setChecked(False)
                self.triangulateBtn.setEnabled(False)
                self.sendPos3DBtn.setChecked(False)
                self.sendPos3DBtn.setEnabled(False)
        self.runDLC.value = self.runDLCBtn.isChecked()
        self.showDLC.value = self.showDLCBtn.isChecked()
        self.showPos2D.value = self.showPos2DBtn.isChecked()
        self.useCyclop.value = self.useCyclopChk.isChecked()
        self.triangulate.value = self.triangulateBtn.isChecked()
        self.sendPos3D.value = self.sendPos3DBtn.isChecked()

    def ShowPos2D(self):
        self.showPos2D.value = self.showPos2DBtn.isChecked()

    def ShowDLC(self):
        self.showDLC.value = self.showDLCBtn.isChecked()

    def UseCyclop(self):
        self.useCyclop.value = self.useCyclopChk.isChecked()

    def Triangulate(self):
        if self.triangulateBtn.isChecked():
            self.sendPos3DBtn.setChecked(True)
            self.sendPos3DBtn.setEnabled(True)
        else:
            self.sendPos3DBtn.setChecked(False)
            self.sendPos3DBtn.setEnabled(False)
        self.triangulate.value = self.triangulateBtn.isChecked()
        self.sendPos3D.value = self.sendPos3DBtn.isChecked()

    def SendPos3D(self):
        self.sendPos3D.value = self.sendPos3DBtn.isChecked()
        self.recvEventPos3DBtn.setEnabled(self.sendPos3D.value)

    def RecvEventPos(self):
        self.recvEventPos.value = self.recvEventPos3DBtn.isChecked()

    def LoadSettings(self):
        self.SendCommandTCPTracking('loadSettings')
        time.sleep(0.5)
        self.UpdateSystemSettingsUI()
        self.UpdateExperimentSettingsUI()

    def SaveResults(self):
        self.saveResults.value = self.saveResultsChk.isChecked()

    def NewRef(self):
        self.SendCommandTCPTracking('newRef')
        self.log.LogText(2, 'UIController: newRef sent')

    def closeEvent(self, event):

        self.Quit()

    def Quit(self):
        # Soft quit
        self.SendCommandTCPTracking('quit')
        self.log.LogText(2, 'UIController: quit sent ("soft" quit when TCP server is up)')

        # Hard quit (if server is down)
        self.stopRequest.value = True
        self.quit.value = True
        self.log.LogText(2, 'UIController: quit and stopAcquisition set to True ("hard" quit when TCP server is down)')

        QApplication.instance().quit

    def SendCommandTCPTracking(self, command, separator='\t'):
        """Send commands to TCP server running on Tracking PC"""

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientSocket:

            # Connects to TCP server on Tracking
            try:
                clientSocket.connect(TCPserverTracking)
            except socket.error as errorMsg:
                self.log.LogText(2, 'SendCommandTCPTracking: Connection error: %s' % errorMsg)
                return 0
            self.log.LogText(2, 'SendCommandTCPTracking: Connected to TCP server')

            # Prepares binary message
            msg = str.encode(command + separator)

            # Sends command
            try:
                clientSocket.sendall(msg)  # Crashes if server is down
            except socket.error as errorMsg:
                self.log.LogText(2, 'SendCommandTCPTracking: Sending error: %s, ignoring' % errorMsg)
                return 0
            self.log.LogText(2, 'SendCommandTCPTracking: cmd=\'%s\' sent' % command)



class Log:
    """Class used to safely log the ongoing of the program (also used for debuging)"""

    def __init__(self, logLevel=int, showTime=True, __output=''):
        """Use output='' for console writing"""

        self.__lock = mp.Lock()          # threading.Lock()
        self.logLevel = logLevel
        self.showTime = showTime
        if showTime:
            self.startTime = time.time_ns()
        if __output != '':
            self.__outToFile = True
            self.__stdoutCopy = sys.stdout
            sys.stdout = open(__output, 'wb')
        else:
            self.__outToFile = False

    def __del__(self):  # Called when destroying object

        del self.__lock
        if self.__outToFile:
            sys.stdout.flush()
            sys.stdout = self.__stdoutCopy

    def LogText(self, level, text):

        if self.logLevel >= level:
            self.__lock.acquire()
            if self.showTime:
                t = float(time.time_ns() - self.startTime) / 1E9
                print('%10.6f\t' % t + '  ' * (level - 1) + text)
            else:
                print('  ' * (level - 1) + text)
            self.__lock.release()


# Starts everything (if this is the main process)
if __name__ == '__main__':

    # Move to TrackingMaster.py directory (if not already)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Multi-processing settings
    # mp.set_start_method('spawn')
    # mp.freeze_support()

    # Start tracking
    myTracking = Tracking()
