from pyheaven import *

if __name__=="__main__":
    args = HeavenArguments.from_parser([
        StrArgumentDescriptor("input",short="i",default=None),
        StrArgumentDescriptor("output",short="o",default=None),
        StrArgumentDescriptor("indent",short="t",default=None),
    ])
    if args.input is not None and args.output is not None:
        SaveJson(LoadJson(args.input,backend='jsonl'),args.output,indent=args.indent)