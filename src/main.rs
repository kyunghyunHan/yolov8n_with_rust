use opencv::{
    core, dnn, highgui,
    prelude::{VideoCaptureTrait, VideoCaptureTraitConst,NetTraitConst},
    videoio, Result,
};
const MODEL:&str =  "./yolov8n.onnx";
fn main() -> Result<()> {
  
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    //frame set
    cap.set(videoio::CAP_PROP_FRAME_WIDTH, 320.0)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 240.0)?;

    //camera is oppend
    if cap.is_opened()? == false {
        println!("{}", " Camera open failed!");
        std::process::exit(0);
    }

    let mut net = dnn::read_net_from_onnx(MODEL)?;
    if net.empty()? {
        println!("{}", "Net Open Failed");
        std::process::exit(0);
    }
    loop{

    }
    Ok(())
}
