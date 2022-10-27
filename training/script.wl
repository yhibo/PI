(*get association of resources, name of local host, and remove local host from available resources*)
hosts = Counts[ReadList[Environment["PBS_NODEFILE"], "String"]];
local = First[StringSplit[Environment["HOSTNAME"],"."]];
hosts[local]--;

(*launch subkernels and connect them to the controlling Wolfram Kernel*)
Needs["SubKernels`RemoteKernels`"];
Map[If[hosts[#] > 0, LaunchKernels[RemoteMachine[#, hosts[#]]]]&, Keys[hosts]];

(* ===== regular Wolfram Language code goes here ===== *)
Print[ {Hola} ]
Print[ {$MachineName, $KernelID} ]
(* ===== end of Wolfram Language program ===== *)

CloseKernels[];
Quit