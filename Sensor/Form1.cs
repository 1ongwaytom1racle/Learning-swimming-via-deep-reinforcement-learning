using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using SeeSharpTools.JY.ArrayUtility;
using SeeSharpTools.JY.DSP.Fundamental;
using SeeSharpTools.JY.File;
using SeeSharpTools.JY.GUI;
using SeeSharpTools.JY.Report;
using JYUSB62401;
using System.IO;
namespace DataToExcel_background
{
    public partial class Form1 : Form
    {

        #region constructor
        public Form1()
        {
            InitializeComponent();
        }
        #endregion


        #region Private Fields
        /// <summary>
        /// AI 任务
        /// </summary>
        JYUSB62401AITask aitask;

        /// <summary>
        /// 输出文档
        /// </summary>
        ExcelReport excel;

        /// <summary>
        /// 记录时间
        /// </summary>
        private double recordLength=60;

        /// <summary>
        /// 采样率
        /// </summary>
        private double samplerate=80;


        /// <summary>
        /// channel number
        /// </summary>
        private int channelnum = 3;
        

        /// <summary>
        /// 读取数据
        /// </summary>
        private double[,] readValue;

        /// <summary>
        /// 预览转置数据数组
        /// </summary>
        double[,] displayValue;

        #endregion

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {

                readValue = new double[(int)(samplerate * recordLength), channelnum];
                displayValue = new double[channelnum, (int)(samplerate * recordLength)];


                //创建任务
                aitask = new JYUSB62401AITask(0);

                //添加每个通道
                for (int i = 0; i < channelnum; i++)
                {
                    aitask.AddChannel(i,ChannelType.Voltage,-5,5,MovingAverageStage.Disable);
                }

                //配置基本参数
                aitask.SampleRate = samplerate;
                aitask.Mode = AIMode.Finite;
                aitask.SamplesToAcquire = (int)(samplerate * recordLength);

                aitask.Start();

                //读取信号
                aitask.ReadData(ref readValue, -1);
                //              ArrayManipulation.Transpose(readValue, ref displayValue);
                //               easyChartX1.Plot(displayValue);

                aitask.Stop();


                excel = new ExcelReport();
                //                excel.Show(); //观看写入数据过程
                excel.WriteArrayToReport("A1", readValue);
                //excel.SaveAs(Directory.GetCurrentDirectory() + @"\dataReport" + DateTime.Now.DayOfYear.ToString() + @"_"
                //+DateTime.Now.Hour.ToString() + @"_" + DateTime.Now.Minute.ToString() + @"_" + DateTime.Now.Second.ToString(), ExcelSaveFormat.csv);
                excel.SaveAs( "C:\\Users\\zhang\\Desktop\\Experiment_Record" + @"\dataReport" + DateTime.Now.DayOfYear.ToString() + @"_"
                  +DateTime.Now.Hour.ToString() + @"_" + DateTime.Now.Minute.ToString() + @"_" + DateTime.Now.Second.ToString(), ExcelSaveFormat.csv);
                
                excel.Close();
                //                CsvHandler.WriteData(readValue);
                //                MessageBox.Show("保存完成");

                // 采集结束之后停机
                aitask.Stop();
                aitask.Channels.Clear();
                Application.Exit();
            }
            catch (JYDriverException ex)
            {
                //驱动错误信息显示
                MessageBox.Show(ex.Message);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            
        }

    }
}
