from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableParallel

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model1 = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

model2 = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate  5 short and simple questions from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the following notes and quiz into a single document \n notes ->{notes} and {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser,
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

texts = """The Chevrolet Corvette (C6) is the sixth generation of the Corvette sports car that was produced by Chevrolet from 2005 to 2013. It is the first Corvette with exposed headlamps (as opposed to hidden headlamps) since the 1962 model. Production variants include the Z06, ZR1, Grand Sport, and 427 Convertible. Racing variants include the C6.R, an American Le Mans Series GT1 championship and 24 Hours of Le Mans GTE-Pro winner.

Overview

Chevrolet Corvette coupe (EU)

Chevrolet Corvette convertible

Chevrolet Corvette C6 interior
The Corvette C6 featured new bodywork with exposed headlamps, revised suspension geometry, a larger passenger compartment, a larger 5,967 cc (364.1 cu in; 6.0 L) V8 engine and a higher level of refinement.[6] It is 5.1 inches (13 cm) shorter than the C5, but the wheelbase was increased by 1.2 inches (3 cm). It is also one inch (2.5 cm) narrower. Like its predecessor, the coupe featured a removable roof panel, either solid in body color or transparent black. The open version featured a power soft top (option not available on the C5). The new GM LS2 engine has a power output of 400 hp (298 kW) at 6,000 rpm and 400 lb⋅ft (542 N⋅m) of torque at 4,400 rpm. Like the C5, the Corvette C6's suspension consisted of independent unequal-length double wishbones with transverse fiberglass mono-leaf springs and optional magnetorheological dampers. Since 2004, a Z51 package has been available, which included a stiffer suspension, larger brakes, a transmission oil cooler, and more.[7] The European version had parking lights in the headlights, headlight washers, side turn signals on the fenders (Grand Sport, ZR1 - different side mirrors with a signal), amber side marker lights, different rear lights with a turn signal and rear fog lights.

With an automatic transmission, the Corvette achieves city/highway fuel economies of 15 / 25 mpg‑US (15.7 / 9.4 L/100 km; 18 / 30 mpg‑imp), while the manual-transmission model returns 16 / 26 mpg‑US (14.7 / 9.0 L/100 km; 19 / 31 mpg‑imp).[8] The Corvette's manual transmission is fitted with Computer Aided Gear Shifting (CAGS), which forces the driver to shift from 1st gear directly to 4th when operating at lower engine speeds. This boosts the EPA's derived fuel economy, allowing the buyer to avoid paying the "gas guzzler" tax. GM badges were added to the car in 2006.

Beginning with the 2008 model year, the Corvette received a new engine, the LS3. With displacement increased to 6,162 cc (376.0 cu in; 6.2 L),[9] power was increased to 430 hp (321 kW) at 6,500 rpm and 424 lb⋅ft (575 N⋅m) of torque at 4,600 rpm, or 436 hp (325 kW) and 428 lb⋅ft (580 N⋅m) with the optional vacuum actuated valve exhaust.[10] The earlier Tremec T56 manual transmission was replaced with the newer TR6060 in the 2008 model year.[11] Manual Corvette models now had improved shift linkage; the automatic model was set up for quicker shifts, and (according to Chevrolet) accelerates from 0 to 60 mph (97 km/h) in 4.0 seconds, faster than any other production automatic Corvette model.[12] From 2008 onward, the steering was revised for improved feel and the wheels were updated to a new five-spoke design.[12] The last Corvette C6, a white 427 convertible, was completed on February 28, 2013.[13]

Z06


Corvette Z06 (below EU spec)

The 7.0 L LS7 V8 engine

The pace car (orange Corvette) leads the field at the 2007 Indianapolis 500.
The Z06 arrived for the 2006 model year as a homologation vehicle in the third quarter of 2005 and is the lightest of all Corvette models. The Z06 was equipped with the largest-displacement small-block engine ever produced, a new 7,011 cc (7.0 L; 427.8 cu in) V8 engine codenamed the LS7. The engine has a power output of 505 hp (512 PS; 377 kW) at 6,300 rpm and 470 lb⋅ft (637 N⋅m) at 4,800 rpm of torque. Until the introduction of the LT6 engine in October 2021, the LS7 was the most powerful production naturally-aspirated engine made by General Motors. In addition to the larger displacement engine, the Corvette Z06 has a dry sump oiling system, and connecting rods made out of titanium alloy. The frame of the Z06 is constructed from aluminum, saving 136 lb (62 kg) over the prior generation C5's steel frame.[14] Other weight saving measures such as balsa wood/carbon fiber composite floors and a magnesium alloy engine cradle were used. The Z06's body differentiates itself from the standard Corvette with its larger front and rear fenders and an intake inlet scoop on the front bumper.[15] The front fenders are constructed with carbon fiber and the rear fenders contain ducts to aid in cooling the rear brakes. The Z06 weighed 3,130 lb (1,420 kg),[16] giving it a weight to power ratio of 6.2 pounds per horsepower (3.8 kg/kW), allowing it to achieve a fuel economy of 15 mpg‑US (16 L/100 km; 18 mpg‑imp) in the city and 24 mpg‑US (9.8 L/100 km; 29 mpg‑imp) on the highway.[17] The Z06 was the official pace car for both the 2006 Daytona 500 and the Indianapolis 500 race. Car and Driver recorded a 0–60 mph (97 km/h) acceleration time of 3.6 seconds and 1/4 mile in 11.7 seconds at 125 mph (201 km/h) in October 2005. The Z06 contains the following enhancements over the standard C6 Corvette:

Hand-built LS7 V8 with dry sump oiling system
Aluminum chassis with fixed roof
Larger/stiffer anti-sway bars
Stiffer springs and shocks
Larger rear spoiler and front splitter
Transmission cooler
Unique front fascia with intake inlet
Wider carbon fiber front and fiberglass rear fenders
Functional front and rear brake ducts for added brake system cooling
Unique light weight forged aluminum 18-inch front and 19-inch rear wheels; painted finish standard and chromed finish optional
Wider 275/35ZR18 tires in the front and wider 325/30ZR19 tires in the rear
Larger 6-piston front and 4-piston rear calipers, including cross-drilled 355 mm (14.0 in) front rotors and 340 mm (13 in) rear rotors
Optional 1LZ, 2LZ, and 3LZ trim packages.
For the 2011 model year, the Z06 Carbon Edition was introduced, which carries over multiple parts from the ZR1 including carbon ceramic brakes, active suspension, and a number of carbon fiber pieces, including the hood and, optionally, the roof panel.[18] Only 500 were planned for production and actual production numbers were much lower, with only about 250 made in total. In 2012 and 2013, these options could be ordered through the Z07 package on the Z06.[19] Additionally, it was only available in Inferno Orange and Supersonic Blue exterior colors, and the interior features unique Z06 Carbon emblem accents.[20][21]

ZR1

"""

result = chain.invoke({"text":{texts}})

print(result)

# graph = chain.get_graph().print_ascii()

# print(graph)


