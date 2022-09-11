import {
  StyleSheet,
  Text,
  View,
  Platform,
  Dimensions,
  LogBox,
} from "react-native";
import React, { useState, useEffect, useRef } from "react";
//camera
import { Camera, CameraType } from "expo-camera";
//tensorflow
import * as tf from "@tensorflow/tfjs";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
//detection
import * as cocoSsd from "@tensorflow-models/coco-ssd";
//classification
import * as mobilenet from "@tensorflow-models/mobilenet";
//canvas
import Canvas from "react-native-canvas";
import { setSyntheticLeadingComments } from "typescript";

export default function App() {
  const [hasPermission, setHasPermission] = useState<null | boolean>(null);
  const TensorCamera = cameraWithTensors(Camera);
  //detection
  const [model, setModel] = useState<cocoSsd.ObjectDetection>();
  let context = useRef<CanvasRenderingContext2D>();
  let canvas = useRef<Canvas>();
  //classification
  const [net, setNet] = useState<mobilenet.MobileNet>();
  const [obj, setObj] = useState([]);
  const [currentObj, setCurrentObj] = useState({});

    let frame = 0;
    const frameRate = 60;

  const { width, height } = Dimensions.get("window");

  const textureDims =
    Platform.OS === "ios"
      ? {
          height: 1920,
          width: 1080,
        }
      : {
          height: 1200,
          width: 1600,
        };

  // const handleCameraStream = (images: any) => {
  //   const loop = async () => {
  //     const nextImageTensor = images.next().value;
  //     if (!model || !nextImageTensor) throw new Error("Error");
  //     model
  //       .detect(nextImageTensor)
  //       .then((prediction) => {
  //         drawRectangle(prediction, nextImageTensor);
  //       })
  //       .catch((error) => {
  //         console.log(error);
  //       });
  //     requestAnimationFrame(loop);
  //   };
  //   loop();
  // };

  const handleCameraStream = (images: IterableIterator<tf.Tensor3D>) => {
    const loop = async () => {
      if (net) {
        if(frame % frameRate === 0){
        const nextImageTensor = images.next().value;
        if (nextImageTensor) {
          const objects = await net.classify(nextImageTensor);
          console.log(objects.map(object => object.className));
          // setObj((prev) => [...prev, objects])
          // setObj((prev) => [...prev, objects.map(object => object.className)]);
          // objects.map(object => object.className)
          tf.dispose([nextImageTensor]);
        }
      }
      frame += 1;
      frame = frame % frameRate;
      }
      requestAnimationFrame(loop);
    }
    loop();
  }

  // const drawRectangle = (predictions: cocoSsd.DetectedObject[], nextImageTensor: any) => {
  //   if (!context.current || !canvas.current) throw new Error("Error");
  //   const scaleWidth = width / nextImageTensor.shape[1];
  //   const scaleHeight = height / nextImageTensor.shape[0];

  //   const flipHorizontal = Platform.OS == "ios" ? false : true;

  //   context.current.clearRect(0, 0, width, height);

  //   for (const prediction of predictions) {
  //     const [x, y, width, height] = prediction.bbox;

  //     const boundingBoxX = flipHorizontal
  //       ? canvas.current.width - x * scaleWidth - width * scaleWidth
  //       : x * scaleWidth;
  //     const boundingBoxY = y * scaleHeight;

  //     context.current.strokeRect(
  //       boundingBoxX,
  //       boundingBoxY,
  //       width * scaleWidth,
  //       height * scaleHeight
  //     );

  //     context.current.strokeText(
  //       prediction.class,
  //       boundingBoxX - 5,
  //       boundingBoxY - 5
  //     );
  //   }
  // };

  // const handleCanvas = async (can: Canvas) => {
  //   if (can) {
  //     can.width = width;
  //     can.height = height;
  //     const ctx : CanvasRenderingContext2D = can.getContext("2d");
  //     ctx.strokeStyle = "red";
  //     ctx.fillStyle = "red";
  //     ctx.lineWidth = 3;

  //     context.current = ctx;
  //     canvas.current = can;
  //   }
  // };

  useEffect(() => {
    (async () => {
      //on load get camera premissions
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === "granted");

      //load model
      await tf.ready();
      tf.getBackend();
      // setModel(await cocoSsd.load());
      // setNet(await mobilenet.load());
      setNet(await mobilenet.load({version: 1, alpha: 0.25}));
      console.log("Model loaded");

      setObj([]);
    })();
  }, []);

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <View style={styles.container}>
      {/* <Camera style={styles.camera} type={Camera.Constants.Type.back}/> */}
      <TensorCamera
        style={styles.camera}
        type={CameraType.back}
        // onReady={
        //   handleCameraStream
        // }
        onReady={handleCameraStream}
        resizeHeight={200}
        resizeWidth={152}
        resizeDepth={3}
        autorender={true}
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        useCustomShadersToResize={true}
      />
      {/* <Canvas style={styles.canvas} ref={handleCanvas} /> */}
      <View style={styles.bottomBar}>
      
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "blue",
  },
  camera: {
    height: "90%",
    width: "100%",
  },
  canvas: {
    position: "absolute",
    marginLeft: "auto",
    marginRight: "auto",
    left: 0,
    right: 0,
    textAlign: "center",
    zindex: 10000,
    width: 640,
    height: 480,
  },
  bottomBar:{
    height: 200,
    width: "100%",
    backgroundColor: "white",
    zindex: 10000,
    position: "absolute",
    top: 550,
  }
});
