??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02unknown8??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
sequential/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namesequential/conv2d/kernel
?
,sequential/conv2d/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d/kernel*&
_output_shapes
:@*
dtype0
?
sequential/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namesequential/conv2d/bias
}
*sequential/conv2d/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d/bias*
_output_shapes
:@*
dtype0
?
sequential/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *+
shared_namesequential/conv2d_1/kernel
?
.sequential/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d_1/kernel*&
_output_shapes
:@ *
dtype0
?
sequential/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namesequential/conv2d_1/bias
?
,sequential/conv2d_1/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d_1/bias*
_output_shapes
: *
dtype0
?
sequential/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namesequential/conv2d_2/kernel
?
.sequential/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d_2/kernel*&
_output_shapes
: *
dtype0
?
sequential/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namesequential/conv2d_2/bias
?
,sequential/conv2d_2/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d_2/bias*
_output_shapes
:*
dtype0
?
$sequential_1/conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$sequential_1/conv2d_transpose/kernel
?
8sequential_1/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOp$sequential_1/conv2d_transpose/kernel*&
_output_shapes
:*
dtype0
?
"sequential_1/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"sequential_1/conv2d_transpose/bias
?
6sequential_1/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOp"sequential_1/conv2d_transpose/bias*
_output_shapes
:*
dtype0
?
&sequential_1/conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&sequential_1/conv2d_transpose_1/kernel
?
:sequential_1/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp&sequential_1/conv2d_transpose_1/kernel*&
_output_shapes
: *
dtype0
?
$sequential_1/conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$sequential_1/conv2d_transpose_1/bias
?
8sequential_1/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOp$sequential_1/conv2d_transpose_1/bias*
_output_shapes
: *
dtype0
?
&sequential_1/conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *7
shared_name(&sequential_1/conv2d_transpose_2/kernel
?
:sequential_1/conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp&sequential_1/conv2d_transpose_2/kernel*&
_output_shapes
:@ *
dtype0
?
$sequential_1/conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$sequential_1/conv2d_transpose_2/bias
?
8sequential_1/conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOp$sequential_1/conv2d_transpose_2/bias*
_output_shapes
:@*
dtype0
?
sequential_1/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namesequential_1/conv2d_3/kernel
?
0sequential_1/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_3/kernel*&
_output_shapes
:@*
dtype0
?
sequential_1/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesequential_1/conv2d_3/bias
?
.sequential_1/conv2d_3/bias/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_3/bias*
_output_shapes
:*
dtype0
?
Adam/sequential/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/sequential/conv2d/kernel/m
?
3Adam/sequential/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
?
Adam/sequential/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/sequential/conv2d/bias/m
?
1Adam/sequential/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/bias/m*
_output_shapes
:@*
dtype0
?
!Adam/sequential/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/sequential/conv2d_1/kernel/m
?
5Adam/sequential/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_1/kernel/m*&
_output_shapes
:@ *
dtype0
?
Adam/sequential/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/sequential/conv2d_1/bias/m
?
3Adam/sequential/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_1/bias/m*
_output_shapes
: *
dtype0
?
!Adam/sequential/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/sequential/conv2d_2/kernel/m
?
5Adam/sequential/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/sequential/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/sequential/conv2d_2/bias/m
?
3Adam/sequential/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_2/bias/m*
_output_shapes
:*
dtype0
?
+Adam/sequential_1/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/sequential_1/conv2d_transpose/kernel/m
?
?Adam/sequential_1/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/sequential_1/conv2d_transpose/kernel/m*&
_output_shapes
:*
dtype0
?
)Adam/sequential_1/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/sequential_1/conv2d_transpose/bias/m
?
=Adam/sequential_1/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOp)Adam/sequential_1/conv2d_transpose/bias/m*
_output_shapes
:*
dtype0
?
-Adam/sequential_1/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/sequential_1/conv2d_transpose_1/kernel/m
?
AAdam/sequential_1/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/sequential_1/conv2d_transpose_1/kernel/m*&
_output_shapes
: *
dtype0
?
+Adam/sequential_1/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/sequential_1/conv2d_transpose_1/bias/m
?
?Adam/sequential_1/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOp+Adam/sequential_1/conv2d_transpose_1/bias/m*
_output_shapes
: *
dtype0
?
-Adam/sequential_1/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *>
shared_name/-Adam/sequential_1/conv2d_transpose_2/kernel/m
?
AAdam/sequential_1/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/sequential_1/conv2d_transpose_2/kernel/m*&
_output_shapes
:@ *
dtype0
?
+Adam/sequential_1/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/sequential_1/conv2d_transpose_2/bias/m
?
?Adam/sequential_1/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOp+Adam/sequential_1/conv2d_transpose_2/bias/m*
_output_shapes
:@*
dtype0
?
#Adam/sequential_1/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/sequential_1/conv2d_3/kernel/m
?
7Adam/sequential_1/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/conv2d_3/kernel/m*&
_output_shapes
:@*
dtype0
?
!Adam/sequential_1/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_1/conv2d_3/bias/m
?
5Adam/sequential_1/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/conv2d_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/sequential/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/sequential/conv2d/kernel/v
?
3Adam/sequential/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/sequential/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/sequential/conv2d/bias/v
?
1Adam/sequential/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/bias/v*
_output_shapes
:@*
dtype0
?
!Adam/sequential/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/sequential/conv2d_1/kernel/v
?
5Adam/sequential/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_1/kernel/v*&
_output_shapes
:@ *
dtype0
?
Adam/sequential/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/sequential/conv2d_1/bias/v
?
3Adam/sequential/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_1/bias/v*
_output_shapes
: *
dtype0
?
!Adam/sequential/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/sequential/conv2d_2/kernel/v
?
5Adam/sequential/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/sequential/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/sequential/conv2d_2/bias/v
?
3Adam/sequential/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_2/bias/v*
_output_shapes
:*
dtype0
?
+Adam/sequential_1/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/sequential_1/conv2d_transpose/kernel/v
?
?Adam/sequential_1/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/sequential_1/conv2d_transpose/kernel/v*&
_output_shapes
:*
dtype0
?
)Adam/sequential_1/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/sequential_1/conv2d_transpose/bias/v
?
=Adam/sequential_1/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOp)Adam/sequential_1/conv2d_transpose/bias/v*
_output_shapes
:*
dtype0
?
-Adam/sequential_1/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/sequential_1/conv2d_transpose_1/kernel/v
?
AAdam/sequential_1/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/sequential_1/conv2d_transpose_1/kernel/v*&
_output_shapes
: *
dtype0
?
+Adam/sequential_1/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/sequential_1/conv2d_transpose_1/bias/v
?
?Adam/sequential_1/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOp+Adam/sequential_1/conv2d_transpose_1/bias/v*
_output_shapes
: *
dtype0
?
-Adam/sequential_1/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *>
shared_name/-Adam/sequential_1/conv2d_transpose_2/kernel/v
?
AAdam/sequential_1/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/sequential_1/conv2d_transpose_2/kernel/v*&
_output_shapes
:@ *
dtype0
?
+Adam/sequential_1/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/sequential_1/conv2d_transpose_2/bias/v
?
?Adam/sequential_1/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOp+Adam/sequential_1/conv2d_transpose_2/bias/v*
_output_shapes
:@*
dtype0
?
#Adam/sequential_1/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/sequential_1/conv2d_3/kernel/v
?
7Adam/sequential_1/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/conv2d_3/kernel/v*&
_output_shapes
:@*
dtype0
?
!Adam/sequential_1/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_1/conv2d_3/bias/v
?
5Adam/sequential_1/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/conv2d_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?H
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?H
value?HB?H B?H
?
encoder
decoder
	optimizer
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
_training_endpoints
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
y

layer-0
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
?
layer-0
layer-1
layer-2
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratempmq mr!ms"mt#mu$mv%mw&mx'my(mz)m{*m|+m}v~v v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?
 
f
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
 
f
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
?
,layer_regularization_losses
-metrics
.non_trainable_variables
	variables
regularization_losses

/layers
trainable_variables
 
h

kernel
bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

 kernel
!bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

"kernel
#bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
*
0
1
 2
!3
"4
#5
 
*
0
1
 2
!3
"4
#5
?
<layer_regularization_losses
=metrics
>non_trainable_variables
	variables
regularization_losses

?layers
trainable_variables
h

$kernel
%bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

&kernel
'bias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
h

(kernel
)bias
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
h

*kernel
+bias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
8
$0
%1
&2
'3
(4
)5
*6
+7
 
8
$0
%1
&2
'3
(4
)5
*6
+7
?
Player_regularization_losses
Qmetrics
Rnon_trainable_variables
	variables
regularization_losses

Slayers
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEsequential/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEsequential/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEsequential/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEsequential/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEsequential/conv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEsequential/conv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$sequential_1/conv2d_transpose/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"sequential_1/conv2d_transpose/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&sequential_1/conv2d_transpose_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$sequential_1/conv2d_transpose_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&sequential_1/conv2d_transpose_2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$sequential_1/conv2d_transpose_2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_1/conv2d_3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEsequential_1/conv2d_3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1

0
1
 

0
1
?
Tlayer_regularization_losses
Unon_trainable_variables
0	variables
Vmetrics
1regularization_losses

Wlayers
2trainable_variables

 0
!1
 

 0
!1
?
Xlayer_regularization_losses
Ynon_trainable_variables
4	variables
Zmetrics
5regularization_losses

[layers
6trainable_variables

"0
#1
 

"0
#1
?
\layer_regularization_losses
]non_trainable_variables
8	variables
^metrics
9regularization_losses

_layers
:trainable_variables
 
 
 


0
1
2

$0
%1
 

$0
%1
?
`layer_regularization_losses
anon_trainable_variables
@	variables
bmetrics
Aregularization_losses

clayers
Btrainable_variables

&0
'1
 

&0
'1
?
dlayer_regularization_losses
enon_trainable_variables
D	variables
fmetrics
Eregularization_losses

glayers
Ftrainable_variables

(0
)1
 

(0
)1
?
hlayer_regularization_losses
inon_trainable_variables
H	variables
jmetrics
Iregularization_losses

klayers
Jtrainable_variables

*0
+1
 

*0
+1
?
llayer_regularization_losses
mnon_trainable_variables
L	variables
nmetrics
Mregularization_losses

olayers
Ntrainable_variables
 
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
wu
VARIABLE_VALUEAdam/sequential/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/sequential/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/sequential/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/sequential/conv2d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/conv2d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/sequential_1/conv2d_transpose/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/sequential_1/conv2d_transpose/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/sequential_1/conv2d_transpose_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/sequential_1/conv2d_transpose_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/sequential_1/conv2d_transpose_2/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/sequential_1/conv2d_transpose_2/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_1/conv2d_3/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_1/conv2d_3/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/sequential/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/sequential/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/sequential/conv2d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/conv2d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/sequential_1/conv2d_transpose/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/sequential_1/conv2d_transpose/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/sequential_1/conv2d_transpose_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/sequential_1/conv2d_transpose_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/sequential_1/conv2d_transpose_2/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/sequential_1/conv2d_transpose_2/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_1/conv2d_3/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_1/conv2d_3/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????00*
dtype0*$
shape:?????????00
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential/conv2d/kernelsequential/conv2d/biassequential/conv2d_1/kernelsequential/conv2d_1/biassequential/conv2d_2/kernelsequential/conv2d_2/bias$sequential_1/conv2d_transpose/kernel"sequential_1/conv2d_transpose/bias&sequential_1/conv2d_transpose_1/kernel$sequential_1/conv2d_transpose_1/bias&sequential_1/conv2d_transpose_2/kernel$sequential_1/conv2d_transpose_2/biassequential_1/conv2d_3/kernelsequential_1/conv2d_3/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_24902
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp,sequential/conv2d/kernel/Read/ReadVariableOp*sequential/conv2d/bias/Read/ReadVariableOp.sequential/conv2d_1/kernel/Read/ReadVariableOp,sequential/conv2d_1/bias/Read/ReadVariableOp.sequential/conv2d_2/kernel/Read/ReadVariableOp,sequential/conv2d_2/bias/Read/ReadVariableOp8sequential_1/conv2d_transpose/kernel/Read/ReadVariableOp6sequential_1/conv2d_transpose/bias/Read/ReadVariableOp:sequential_1/conv2d_transpose_1/kernel/Read/ReadVariableOp8sequential_1/conv2d_transpose_1/bias/Read/ReadVariableOp:sequential_1/conv2d_transpose_2/kernel/Read/ReadVariableOp8sequential_1/conv2d_transpose_2/bias/Read/ReadVariableOp0sequential_1/conv2d_3/kernel/Read/ReadVariableOp.sequential_1/conv2d_3/bias/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/m/Read/ReadVariableOp1Adam/sequential/conv2d/bias/m/Read/ReadVariableOp5Adam/sequential/conv2d_1/kernel/m/Read/ReadVariableOp3Adam/sequential/conv2d_1/bias/m/Read/ReadVariableOp5Adam/sequential/conv2d_2/kernel/m/Read/ReadVariableOp3Adam/sequential/conv2d_2/bias/m/Read/ReadVariableOp?Adam/sequential_1/conv2d_transpose/kernel/m/Read/ReadVariableOp=Adam/sequential_1/conv2d_transpose/bias/m/Read/ReadVariableOpAAdam/sequential_1/conv2d_transpose_1/kernel/m/Read/ReadVariableOp?Adam/sequential_1/conv2d_transpose_1/bias/m/Read/ReadVariableOpAAdam/sequential_1/conv2d_transpose_2/kernel/m/Read/ReadVariableOp?Adam/sequential_1/conv2d_transpose_2/bias/m/Read/ReadVariableOp7Adam/sequential_1/conv2d_3/kernel/m/Read/ReadVariableOp5Adam/sequential_1/conv2d_3/bias/m/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/v/Read/ReadVariableOp1Adam/sequential/conv2d/bias/v/Read/ReadVariableOp5Adam/sequential/conv2d_1/kernel/v/Read/ReadVariableOp3Adam/sequential/conv2d_1/bias/v/Read/ReadVariableOp5Adam/sequential/conv2d_2/kernel/v/Read/ReadVariableOp3Adam/sequential/conv2d_2/bias/v/Read/ReadVariableOp?Adam/sequential_1/conv2d_transpose/kernel/v/Read/ReadVariableOp=Adam/sequential_1/conv2d_transpose/bias/v/Read/ReadVariableOpAAdam/sequential_1/conv2d_transpose_1/kernel/v/Read/ReadVariableOp?Adam/sequential_1/conv2d_transpose_1/bias/v/Read/ReadVariableOpAAdam/sequential_1/conv2d_transpose_2/kernel/v/Read/ReadVariableOp?Adam/sequential_1/conv2d_transpose_2/bias/v/Read/ReadVariableOp7Adam/sequential_1/conv2d_3/kernel/v/Read/ReadVariableOp5Adam/sequential_1/conv2d_3/bias/v/Read/ReadVariableOpConst*<
Tin5
321	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_25649
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratesequential/conv2d/kernelsequential/conv2d/biassequential/conv2d_1/kernelsequential/conv2d_1/biassequential/conv2d_2/kernelsequential/conv2d_2/bias$sequential_1/conv2d_transpose/kernel"sequential_1/conv2d_transpose/bias&sequential_1/conv2d_transpose_1/kernel$sequential_1/conv2d_transpose_1/bias&sequential_1/conv2d_transpose_2/kernel$sequential_1/conv2d_transpose_2/biassequential_1/conv2d_3/kernelsequential_1/conv2d_3/biasAdam/sequential/conv2d/kernel/mAdam/sequential/conv2d/bias/m!Adam/sequential/conv2d_1/kernel/mAdam/sequential/conv2d_1/bias/m!Adam/sequential/conv2d_2/kernel/mAdam/sequential/conv2d_2/bias/m+Adam/sequential_1/conv2d_transpose/kernel/m)Adam/sequential_1/conv2d_transpose/bias/m-Adam/sequential_1/conv2d_transpose_1/kernel/m+Adam/sequential_1/conv2d_transpose_1/bias/m-Adam/sequential_1/conv2d_transpose_2/kernel/m+Adam/sequential_1/conv2d_transpose_2/bias/m#Adam/sequential_1/conv2d_3/kernel/m!Adam/sequential_1/conv2d_3/bias/mAdam/sequential/conv2d/kernel/vAdam/sequential/conv2d/bias/v!Adam/sequential/conv2d_1/kernel/vAdam/sequential/conv2d_1/bias/v!Adam/sequential/conv2d_2/kernel/vAdam/sequential/conv2d_2/bias/v+Adam/sequential_1/conv2d_transpose/kernel/v)Adam/sequential_1/conv2d_transpose/bias/v-Adam/sequential_1/conv2d_transpose_1/kernel/v+Adam/sequential_1/conv2d_transpose_1/bias/v-Adam/sequential_1/conv2d_transpose_2/kernel/v+Adam/sequential_1/conv2d_transpose_2/bias/v#Adam/sequential_1/conv2d_3/kernel/v!Adam/sequential_1/conv2d_3/bias/v*;
Tin4
220*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_25802??
Ŋ
?
 __inference__wrapped_model_24345
input_1B
>convolutional_sequential_conv2d_conv2d_readvariableop_resourceC
?convolutional_sequential_conv2d_biasadd_readvariableop_resourceD
@convolutional_sequential_conv2d_1_conv2d_readvariableop_resourceE
Aconvolutional_sequential_conv2d_1_biasadd_readvariableop_resourceD
@convolutional_sequential_conv2d_2_conv2d_readvariableop_resourceE
Aconvolutional_sequential_conv2d_2_biasadd_readvariableop_resourceX
Tconvolutional_sequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceO
Kconvolutional_sequential_1_conv2d_transpose_biasadd_readvariableop_resourceZ
Vconvolutional_sequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceQ
Mconvolutional_sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceZ
Vconvolutional_sequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceQ
Mconvolutional_sequential_1_conv2d_transpose_2_biasadd_readvariableop_resourceF
Bconvolutional_sequential_1_conv2d_3_conv2d_readvariableop_resourceG
Cconvolutional_sequential_1_conv2d_3_biasadd_readvariableop_resource
identity??6convolutional/sequential/conv2d/BiasAdd/ReadVariableOp?5convolutional/sequential/conv2d/Conv2D/ReadVariableOp?8convolutional/sequential/conv2d_1/BiasAdd/ReadVariableOp?7convolutional/sequential/conv2d_1/Conv2D/ReadVariableOp?8convolutional/sequential/conv2d_2/BiasAdd/ReadVariableOp?7convolutional/sequential/conv2d_2/Conv2D/ReadVariableOp?:convolutional/sequential_1/conv2d_3/BiasAdd/ReadVariableOp?9convolutional/sequential_1/conv2d_3/Conv2D/ReadVariableOp?Bconvolutional/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?Kconvolutional/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?Dconvolutional/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?Mconvolutional/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?Dconvolutional/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?Mconvolutional/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
5convolutional/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp>convolutional_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype027
5convolutional/sequential/conv2d/Conv2D/ReadVariableOp?
&convolutional/sequential/conv2d/Conv2DConv2Dinput_1=convolutional/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2(
&convolutional/sequential/conv2d/Conv2D?
6convolutional/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp?convolutional_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6convolutional/sequential/conv2d/BiasAdd/ReadVariableOp?
'convolutional/sequential/conv2d/BiasAddBiasAdd/convolutional/sequential/conv2d/Conv2D:output:0>convolutional/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2)
'convolutional/sequential/conv2d/BiasAdd?
$convolutional/sequential/conv2d/ReluRelu0convolutional/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2&
$convolutional/sequential/conv2d/Relu?
7convolutional/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@convolutional_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype029
7convolutional/sequential/conv2d_1/Conv2D/ReadVariableOp?
(convolutional/sequential/conv2d_1/Conv2DConv2D2convolutional/sequential/conv2d/Relu:activations:0?convolutional/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2*
(convolutional/sequential/conv2d_1/Conv2D?
8convolutional/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpAconvolutional_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8convolutional/sequential/conv2d_1/BiasAdd/ReadVariableOp?
)convolutional/sequential/conv2d_1/BiasAddBiasAdd1convolutional/sequential/conv2d_1/Conv2D:output:0@convolutional/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2+
)convolutional/sequential/conv2d_1/BiasAdd?
&convolutional/sequential/conv2d_1/ReluRelu2convolutional/sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2(
&convolutional/sequential/conv2d_1/Relu?
7convolutional/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@convolutional_sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype029
7convolutional/sequential/conv2d_2/Conv2D/ReadVariableOp?
(convolutional/sequential/conv2d_2/Conv2DConv2D4convolutional/sequential/conv2d_1/Relu:activations:0?convolutional/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2*
(convolutional/sequential/conv2d_2/Conv2D?
8convolutional/sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpAconvolutional_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8convolutional/sequential/conv2d_2/BiasAdd/ReadVariableOp?
)convolutional/sequential/conv2d_2/BiasAddBiasAdd1convolutional/sequential/conv2d_2/Conv2D:output:0@convolutional/sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2+
)convolutional/sequential/conv2d_2/BiasAdd?
&convolutional/sequential/conv2d_2/ReluRelu2convolutional/sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2(
&convolutional/sequential/conv2d_2/Relu?
1convolutional/sequential_1/conv2d_transpose/ShapeShape4convolutional/sequential/conv2d_2/Relu:activations:0*
T0*
_output_shapes
:23
1convolutional/sequential_1/conv2d_transpose/Shape?
?convolutional/sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?convolutional/sequential_1/conv2d_transpose/strided_slice/stack?
Aconvolutional/sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Aconvolutional/sequential_1/conv2d_transpose/strided_slice/stack_1?
Aconvolutional/sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Aconvolutional/sequential_1/conv2d_transpose/strided_slice/stack_2?
9convolutional/sequential_1/conv2d_transpose/strided_sliceStridedSlice:convolutional/sequential_1/conv2d_transpose/Shape:output:0Hconvolutional/sequential_1/conv2d_transpose/strided_slice/stack:output:0Jconvolutional/sequential_1/conv2d_transpose/strided_slice/stack_1:output:0Jconvolutional/sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9convolutional/sequential_1/conv2d_transpose/strided_slice?
Aconvolutional/sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
Aconvolutional/sequential_1/conv2d_transpose/strided_slice_1/stack?
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_1/stack_1?
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_1/stack_2?
;convolutional/sequential_1/conv2d_transpose/strided_slice_1StridedSlice:convolutional/sequential_1/conv2d_transpose/Shape:output:0Jconvolutional/sequential_1/conv2d_transpose/strided_slice_1/stack:output:0Lconvolutional/sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0Lconvolutional/sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;convolutional/sequential_1/conv2d_transpose/strided_slice_1?
Aconvolutional/sequential_1/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
Aconvolutional/sequential_1/conv2d_transpose/strided_slice_2/stack?
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_2/stack_1?
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_2/stack_2?
;convolutional/sequential_1/conv2d_transpose/strided_slice_2StridedSlice:convolutional/sequential_1/conv2d_transpose/Shape:output:0Jconvolutional/sequential_1/conv2d_transpose/strided_slice_2/stack:output:0Lconvolutional/sequential_1/conv2d_transpose/strided_slice_2/stack_1:output:0Lconvolutional/sequential_1/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;convolutional/sequential_1/conv2d_transpose/strided_slice_2?
1convolutional/sequential_1/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :23
1convolutional/sequential_1/conv2d_transpose/mul/y?
/convolutional/sequential_1/conv2d_transpose/mulMulDconvolutional/sequential_1/conv2d_transpose/strided_slice_1:output:0:convolutional/sequential_1/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 21
/convolutional/sequential_1/conv2d_transpose/mul?
3convolutional/sequential_1/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :25
3convolutional/sequential_1/conv2d_transpose/mul_1/y?
1convolutional/sequential_1/conv2d_transpose/mul_1MulDconvolutional/sequential_1/conv2d_transpose/strided_slice_2:output:0<convolutional/sequential_1/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 23
1convolutional/sequential_1/conv2d_transpose/mul_1?
3convolutional/sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :25
3convolutional/sequential_1/conv2d_transpose/stack/3?
1convolutional/sequential_1/conv2d_transpose/stackPackBconvolutional/sequential_1/conv2d_transpose/strided_slice:output:03convolutional/sequential_1/conv2d_transpose/mul:z:05convolutional/sequential_1/conv2d_transpose/mul_1:z:0<convolutional/sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:23
1convolutional/sequential_1/conv2d_transpose/stack?
Aconvolutional/sequential_1/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Aconvolutional/sequential_1/conv2d_transpose/strided_slice_3/stack?
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_3/stack_1?
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose/strided_slice_3/stack_2?
;convolutional/sequential_1/conv2d_transpose/strided_slice_3StridedSlice:convolutional/sequential_1/conv2d_transpose/stack:output:0Jconvolutional/sequential_1/conv2d_transpose/strided_slice_3/stack:output:0Lconvolutional/sequential_1/conv2d_transpose/strided_slice_3/stack_1:output:0Lconvolutional/sequential_1/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;convolutional/sequential_1/conv2d_transpose/strided_slice_3?
Kconvolutional/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpTconvolutional_sequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02M
Kconvolutional/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
<convolutional/sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput:convolutional/sequential_1/conv2d_transpose/stack:output:0Sconvolutional/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:04convolutional/sequential/conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2>
<convolutional/sequential_1/conv2d_transpose/conv2d_transpose?
Bconvolutional/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpKconvolutional_sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bconvolutional/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
3convolutional/sequential_1/conv2d_transpose/BiasAddBiasAddEconvolutional/sequential_1/conv2d_transpose/conv2d_transpose:output:0Jconvolutional/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????25
3convolutional/sequential_1/conv2d_transpose/BiasAdd?
0convolutional/sequential_1/conv2d_transpose/ReluRelu<convolutional/sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22
0convolutional/sequential_1/conv2d_transpose/Relu?
3convolutional/sequential_1/conv2d_transpose_1/ShapeShape>convolutional/sequential_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:25
3convolutional/sequential_1/conv2d_transpose_1/Shape?
Aconvolutional/sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Aconvolutional/sequential_1/conv2d_transpose_1/strided_slice/stack?
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice/stack_1?
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice/stack_2?
;convolutional/sequential_1/conv2d_transpose_1/strided_sliceStridedSlice<convolutional/sequential_1/conv2d_transpose_1/Shape:output:0Jconvolutional/sequential_1/conv2d_transpose_1/strided_slice/stack:output:0Lconvolutional/sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0Lconvolutional/sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;convolutional/sequential_1/conv2d_transpose_1/strided_slice?
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice_1/stack?
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
=convolutional/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice<convolutional/sequential_1/conv2d_transpose_1/Shape:output:0Lconvolutional/sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0Nconvolutional/sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Nconvolutional/sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=convolutional/sequential_1/conv2d_transpose_1/strided_slice_1?
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice_2/stack?
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_2/stack_1?
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_2/stack_2?
=convolutional/sequential_1/conv2d_transpose_1/strided_slice_2StridedSlice<convolutional/sequential_1/conv2d_transpose_1/Shape:output:0Lconvolutional/sequential_1/conv2d_transpose_1/strided_slice_2/stack:output:0Nconvolutional/sequential_1/conv2d_transpose_1/strided_slice_2/stack_1:output:0Nconvolutional/sequential_1/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=convolutional/sequential_1/conv2d_transpose_1/strided_slice_2?
3convolutional/sequential_1/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :25
3convolutional/sequential_1/conv2d_transpose_1/mul/y?
1convolutional/sequential_1/conv2d_transpose_1/mulMulFconvolutional/sequential_1/conv2d_transpose_1/strided_slice_1:output:0<convolutional/sequential_1/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 23
1convolutional/sequential_1/conv2d_transpose_1/mul?
5convolutional/sequential_1/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :27
5convolutional/sequential_1/conv2d_transpose_1/mul_1/y?
3convolutional/sequential_1/conv2d_transpose_1/mul_1MulFconvolutional/sequential_1/conv2d_transpose_1/strided_slice_2:output:0>convolutional/sequential_1/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 25
3convolutional/sequential_1/conv2d_transpose_1/mul_1?
5convolutional/sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 27
5convolutional/sequential_1/conv2d_transpose_1/stack/3?
3convolutional/sequential_1/conv2d_transpose_1/stackPackDconvolutional/sequential_1/conv2d_transpose_1/strided_slice:output:05convolutional/sequential_1/conv2d_transpose_1/mul:z:07convolutional/sequential_1/conv2d_transpose_1/mul_1:z:0>convolutional/sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:25
3convolutional/sequential_1/conv2d_transpose_1/stack?
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cconvolutional/sequential_1/conv2d_transpose_1/strided_slice_3/stack?
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_3/stack_1?
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_1/strided_slice_3/stack_2?
=convolutional/sequential_1/conv2d_transpose_1/strided_slice_3StridedSlice<convolutional/sequential_1/conv2d_transpose_1/stack:output:0Lconvolutional/sequential_1/conv2d_transpose_1/strided_slice_3/stack:output:0Nconvolutional/sequential_1/conv2d_transpose_1/strided_slice_3/stack_1:output:0Nconvolutional/sequential_1/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=convolutional/sequential_1/conv2d_transpose_1/strided_slice_3?
Mconvolutional/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpVconvolutional_sequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02O
Mconvolutional/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
>convolutional/sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput<convolutional/sequential_1/conv2d_transpose_1/stack:output:0Uconvolutional/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0>convolutional/sequential_1/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2@
>convolutional/sequential_1/conv2d_transpose_1/conv2d_transpose?
Dconvolutional/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpMconvolutional_sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02F
Dconvolutional/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
5convolutional/sequential_1/conv2d_transpose_1/BiasAddBiasAddGconvolutional/sequential_1/conv2d_transpose_1/conv2d_transpose:output:0Lconvolutional/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 27
5convolutional/sequential_1/conv2d_transpose_1/BiasAdd?
2convolutional/sequential_1/conv2d_transpose_1/ReluRelu>convolutional/sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 24
2convolutional/sequential_1/conv2d_transpose_1/Relu?
3convolutional/sequential_1/conv2d_transpose_2/ShapeShape@convolutional/sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:25
3convolutional/sequential_1/conv2d_transpose_2/Shape?
Aconvolutional/sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Aconvolutional/sequential_1/conv2d_transpose_2/strided_slice/stack?
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice/stack_1?
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice/stack_2?
;convolutional/sequential_1/conv2d_transpose_2/strided_sliceStridedSlice<convolutional/sequential_1/conv2d_transpose_2/Shape:output:0Jconvolutional/sequential_1/conv2d_transpose_2/strided_slice/stack:output:0Lconvolutional/sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0Lconvolutional/sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;convolutional/sequential_1/conv2d_transpose_2/strided_slice?
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice_1/stack?
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
=convolutional/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice<convolutional/sequential_1/conv2d_transpose_2/Shape:output:0Lconvolutional/sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0Nconvolutional/sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0Nconvolutional/sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=convolutional/sequential_1/conv2d_transpose_2/strided_slice_1?
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice_2/stack?
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_2/stack_1?
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_2/stack_2?
=convolutional/sequential_1/conv2d_transpose_2/strided_slice_2StridedSlice<convolutional/sequential_1/conv2d_transpose_2/Shape:output:0Lconvolutional/sequential_1/conv2d_transpose_2/strided_slice_2/stack:output:0Nconvolutional/sequential_1/conv2d_transpose_2/strided_slice_2/stack_1:output:0Nconvolutional/sequential_1/conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=convolutional/sequential_1/conv2d_transpose_2/strided_slice_2?
3convolutional/sequential_1/conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :25
3convolutional/sequential_1/conv2d_transpose_2/mul/y?
1convolutional/sequential_1/conv2d_transpose_2/mulMulFconvolutional/sequential_1/conv2d_transpose_2/strided_slice_1:output:0<convolutional/sequential_1/conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 23
1convolutional/sequential_1/conv2d_transpose_2/mul?
5convolutional/sequential_1/conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :27
5convolutional/sequential_1/conv2d_transpose_2/mul_1/y?
3convolutional/sequential_1/conv2d_transpose_2/mul_1MulFconvolutional/sequential_1/conv2d_transpose_2/strided_slice_2:output:0>convolutional/sequential_1/conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 25
3convolutional/sequential_1/conv2d_transpose_2/mul_1?
5convolutional/sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@27
5convolutional/sequential_1/conv2d_transpose_2/stack/3?
3convolutional/sequential_1/conv2d_transpose_2/stackPackDconvolutional/sequential_1/conv2d_transpose_2/strided_slice:output:05convolutional/sequential_1/conv2d_transpose_2/mul:z:07convolutional/sequential_1/conv2d_transpose_2/mul_1:z:0>convolutional/sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:25
3convolutional/sequential_1/conv2d_transpose_2/stack?
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cconvolutional/sequential_1/conv2d_transpose_2/strided_slice_3/stack?
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_3/stack_1?
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Econvolutional/sequential_1/conv2d_transpose_2/strided_slice_3/stack_2?
=convolutional/sequential_1/conv2d_transpose_2/strided_slice_3StridedSlice<convolutional/sequential_1/conv2d_transpose_2/stack:output:0Lconvolutional/sequential_1/conv2d_transpose_2/strided_slice_3/stack:output:0Nconvolutional/sequential_1/conv2d_transpose_2/strided_slice_3/stack_1:output:0Nconvolutional/sequential_1/conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=convolutional/sequential_1/conv2d_transpose_2/strided_slice_3?
Mconvolutional/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpVconvolutional_sequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02O
Mconvolutional/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
>convolutional/sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput<convolutional/sequential_1/conv2d_transpose_2/stack:output:0Uconvolutional/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0@convolutional/sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2@
>convolutional/sequential_1/conv2d_transpose_2/conv2d_transpose?
Dconvolutional/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpMconvolutional_sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dconvolutional/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
5convolutional/sequential_1/conv2d_transpose_2/BiasAddBiasAddGconvolutional/sequential_1/conv2d_transpose_2/conv2d_transpose:output:0Lconvolutional/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@27
5convolutional/sequential_1/conv2d_transpose_2/BiasAdd?
2convolutional/sequential_1/conv2d_transpose_2/ReluRelu>convolutional/sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@24
2convolutional/sequential_1/conv2d_transpose_2/Relu?
9convolutional/sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOpBconvolutional_sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02;
9convolutional/sequential_1/conv2d_3/Conv2D/ReadVariableOp?
*convolutional/sequential_1/conv2d_3/Conv2DConv2D@convolutional/sequential_1/conv2d_transpose_2/Relu:activations:0Aconvolutional/sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
2,
*convolutional/sequential_1/conv2d_3/Conv2D?
:convolutional/sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpCconvolutional_sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:convolutional/sequential_1/conv2d_3/BiasAdd/ReadVariableOp?
+convolutional/sequential_1/conv2d_3/BiasAddBiasAdd3convolutional/sequential_1/conv2d_3/Conv2D:output:0Bconvolutional/sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????002-
+convolutional/sequential_1/conv2d_3/BiasAdd?
+convolutional/sequential_1/conv2d_3/SigmoidSigmoid4convolutional/sequential_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????002-
+convolutional/sequential_1/conv2d_3/Sigmoid?
IdentityIdentity/convolutional/sequential_1/conv2d_3/Sigmoid:y:07^convolutional/sequential/conv2d/BiasAdd/ReadVariableOp6^convolutional/sequential/conv2d/Conv2D/ReadVariableOp9^convolutional/sequential/conv2d_1/BiasAdd/ReadVariableOp8^convolutional/sequential/conv2d_1/Conv2D/ReadVariableOp9^convolutional/sequential/conv2d_2/BiasAdd/ReadVariableOp8^convolutional/sequential/conv2d_2/Conv2D/ReadVariableOp;^convolutional/sequential_1/conv2d_3/BiasAdd/ReadVariableOp:^convolutional/sequential_1/conv2d_3/Conv2D/ReadVariableOpC^convolutional/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpL^convolutional/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpE^convolutional/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpN^convolutional/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpE^convolutional/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpN^convolutional/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????002

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::2p
6convolutional/sequential/conv2d/BiasAdd/ReadVariableOp6convolutional/sequential/conv2d/BiasAdd/ReadVariableOp2n
5convolutional/sequential/conv2d/Conv2D/ReadVariableOp5convolutional/sequential/conv2d/Conv2D/ReadVariableOp2t
8convolutional/sequential/conv2d_1/BiasAdd/ReadVariableOp8convolutional/sequential/conv2d_1/BiasAdd/ReadVariableOp2r
7convolutional/sequential/conv2d_1/Conv2D/ReadVariableOp7convolutional/sequential/conv2d_1/Conv2D/ReadVariableOp2t
8convolutional/sequential/conv2d_2/BiasAdd/ReadVariableOp8convolutional/sequential/conv2d_2/BiasAdd/ReadVariableOp2r
7convolutional/sequential/conv2d_2/Conv2D/ReadVariableOp7convolutional/sequential/conv2d_2/Conv2D/ReadVariableOp2x
:convolutional/sequential_1/conv2d_3/BiasAdd/ReadVariableOp:convolutional/sequential_1/conv2d_3/BiasAdd/ReadVariableOp2v
9convolutional/sequential_1/conv2d_3/Conv2D/ReadVariableOp9convolutional/sequential_1/conv2d_3/Conv2D/ReadVariableOp2?
Bconvolutional/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpBconvolutional/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2?
Kconvolutional/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpKconvolutional/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2?
Dconvolutional/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpDconvolutional/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
Mconvolutional/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpMconvolutional/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2?
Dconvolutional/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpDconvolutional/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
Mconvolutional/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpMconvolutional/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:' #
!
_user_specified_name	input_1
?
?
H__inference_convolutional_layer_call_and_return_conditional_losses_24815
input_1-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5/
+sequential_1_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_7/
+sequential_1_statefulpartitionedcall_args_8
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244752$
"sequential/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5+sequential_1_statefulpartitionedcall_args_6+sequential_1_statefulpartitionedcall_args_7+sequential_1_statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247152&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?$
?
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_24605

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_24686

inputs3
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2
identity?? conv2d_3/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputs/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_245192*
(conv2d_transpose/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_245622,
*conv2d_transpose_1/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_246052,
*conv2d_transpose_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_246262"
 conv2d_3/StatefulPartitionedCall?
IdentityIdentity)conv2d_3/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_25234

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Relu?
IdentityIdentityconv2d_2/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_24422
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_243582 
conv2d/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_243792"
 conv2d_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_244002"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24379

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
-__inference_convolutional_layer_call_fn_24855
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_convolutional_layer_call_and_return_conditional_losses_248382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
-__inference_convolutional_layer_call_fn_24874
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_convolutional_layer_call_and_return_conditional_losses_248382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?

?
,__inference_sequential_1_layer_call_fn_24697
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_246862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?

?
,__inference_sequential_1_layer_call_fn_25471

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_246862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_25245

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_24519

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
-__inference_convolutional_layer_call_fn_25184
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_convolutional_layer_call_and_return_conditional_losses_248382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
?_
?
__inference__traced_save_25649
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop7
3savev2_sequential_conv2d_kernel_read_readvariableop5
1savev2_sequential_conv2d_bias_read_readvariableop9
5savev2_sequential_conv2d_1_kernel_read_readvariableop7
3savev2_sequential_conv2d_1_bias_read_readvariableop9
5savev2_sequential_conv2d_2_kernel_read_readvariableop7
3savev2_sequential_conv2d_2_bias_read_readvariableopC
?savev2_sequential_1_conv2d_transpose_kernel_read_readvariableopA
=savev2_sequential_1_conv2d_transpose_bias_read_readvariableopE
Asavev2_sequential_1_conv2d_transpose_1_kernel_read_readvariableopC
?savev2_sequential_1_conv2d_transpose_1_bias_read_readvariableopE
Asavev2_sequential_1_conv2d_transpose_2_kernel_read_readvariableopC
?savev2_sequential_1_conv2d_transpose_2_bias_read_readvariableop;
7savev2_sequential_1_conv2d_3_kernel_read_readvariableop9
5savev2_sequential_1_conv2d_3_bias_read_readvariableop>
:savev2_adam_sequential_conv2d_kernel_m_read_readvariableop<
8savev2_adam_sequential_conv2d_bias_m_read_readvariableop@
<savev2_adam_sequential_conv2d_1_kernel_m_read_readvariableop>
:savev2_adam_sequential_conv2d_1_bias_m_read_readvariableop@
<savev2_adam_sequential_conv2d_2_kernel_m_read_readvariableop>
:savev2_adam_sequential_conv2d_2_bias_m_read_readvariableopJ
Fsavev2_adam_sequential_1_conv2d_transpose_kernel_m_read_readvariableopH
Dsavev2_adam_sequential_1_conv2d_transpose_bias_m_read_readvariableopL
Hsavev2_adam_sequential_1_conv2d_transpose_1_kernel_m_read_readvariableopJ
Fsavev2_adam_sequential_1_conv2d_transpose_1_bias_m_read_readvariableopL
Hsavev2_adam_sequential_1_conv2d_transpose_2_kernel_m_read_readvariableopJ
Fsavev2_adam_sequential_1_conv2d_transpose_2_bias_m_read_readvariableopB
>savev2_adam_sequential_1_conv2d_3_kernel_m_read_readvariableop@
<savev2_adam_sequential_1_conv2d_3_bias_m_read_readvariableop>
:savev2_adam_sequential_conv2d_kernel_v_read_readvariableop<
8savev2_adam_sequential_conv2d_bias_v_read_readvariableop@
<savev2_adam_sequential_conv2d_1_kernel_v_read_readvariableop>
:savev2_adam_sequential_conv2d_1_bias_v_read_readvariableop@
<savev2_adam_sequential_conv2d_2_kernel_v_read_readvariableop>
:savev2_adam_sequential_conv2d_2_bias_v_read_readvariableopJ
Fsavev2_adam_sequential_1_conv2d_transpose_kernel_v_read_readvariableopH
Dsavev2_adam_sequential_1_conv2d_transpose_bias_v_read_readvariableopL
Hsavev2_adam_sequential_1_conv2d_transpose_1_kernel_v_read_readvariableopJ
Fsavev2_adam_sequential_1_conv2d_transpose_1_bias_v_read_readvariableopL
Hsavev2_adam_sequential_1_conv2d_transpose_2_kernel_v_read_readvariableopJ
Fsavev2_adam_sequential_1_conv2d_transpose_2_bias_v_read_readvariableopB
>savev2_adam_sequential_1_conv2d_3_kernel_v_read_readvariableop@
<savev2_adam_sequential_1_conv2d_3_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3d92d9728c8b44eca1b5215a263dfaa6/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*?
value?B?/B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop3savev2_sequential_conv2d_kernel_read_readvariableop1savev2_sequential_conv2d_bias_read_readvariableop5savev2_sequential_conv2d_1_kernel_read_readvariableop3savev2_sequential_conv2d_1_bias_read_readvariableop5savev2_sequential_conv2d_2_kernel_read_readvariableop3savev2_sequential_conv2d_2_bias_read_readvariableop?savev2_sequential_1_conv2d_transpose_kernel_read_readvariableop=savev2_sequential_1_conv2d_transpose_bias_read_readvariableopAsavev2_sequential_1_conv2d_transpose_1_kernel_read_readvariableop?savev2_sequential_1_conv2d_transpose_1_bias_read_readvariableopAsavev2_sequential_1_conv2d_transpose_2_kernel_read_readvariableop?savev2_sequential_1_conv2d_transpose_2_bias_read_readvariableop7savev2_sequential_1_conv2d_3_kernel_read_readvariableop5savev2_sequential_1_conv2d_3_bias_read_readvariableop:savev2_adam_sequential_conv2d_kernel_m_read_readvariableop8savev2_adam_sequential_conv2d_bias_m_read_readvariableop<savev2_adam_sequential_conv2d_1_kernel_m_read_readvariableop:savev2_adam_sequential_conv2d_1_bias_m_read_readvariableop<savev2_adam_sequential_conv2d_2_kernel_m_read_readvariableop:savev2_adam_sequential_conv2d_2_bias_m_read_readvariableopFsavev2_adam_sequential_1_conv2d_transpose_kernel_m_read_readvariableopDsavev2_adam_sequential_1_conv2d_transpose_bias_m_read_readvariableopHsavev2_adam_sequential_1_conv2d_transpose_1_kernel_m_read_readvariableopFsavev2_adam_sequential_1_conv2d_transpose_1_bias_m_read_readvariableopHsavev2_adam_sequential_1_conv2d_transpose_2_kernel_m_read_readvariableopFsavev2_adam_sequential_1_conv2d_transpose_2_bias_m_read_readvariableop>savev2_adam_sequential_1_conv2d_3_kernel_m_read_readvariableop<savev2_adam_sequential_1_conv2d_3_bias_m_read_readvariableop:savev2_adam_sequential_conv2d_kernel_v_read_readvariableop8savev2_adam_sequential_conv2d_bias_v_read_readvariableop<savev2_adam_sequential_conv2d_1_kernel_v_read_readvariableop:savev2_adam_sequential_conv2d_1_bias_v_read_readvariableop<savev2_adam_sequential_conv2d_2_kernel_v_read_readvariableop:savev2_adam_sequential_conv2d_2_bias_v_read_readvariableopFsavev2_adam_sequential_1_conv2d_transpose_kernel_v_read_readvariableopDsavev2_adam_sequential_1_conv2d_transpose_bias_v_read_readvariableopHsavev2_adam_sequential_1_conv2d_transpose_1_kernel_v_read_readvariableopFsavev2_adam_sequential_1_conv2d_transpose_1_bias_v_read_readvariableopHsavev2_adam_sequential_1_conv2d_transpose_2_kernel_v_read_readvariableopFsavev2_adam_sequential_1_conv2d_transpose_2_bias_v_read_readvariableop>savev2_adam_sequential_1_conv2d_3_kernel_v_read_readvariableop<savev2_adam_sequential_1_conv2d_3_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :@:@:@ : : :::: : :@ :@:@::@:@:@ : : :::: : :@ :@:@::@:@:@ : : :::: : :@ :@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?	
?
*__inference_sequential_layer_call_fn_24484
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_24667
input_13
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2
identity?? conv2d_3/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinput_1/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_245192*
(conv2d_transpose/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_245622,
*conv2d_transpose_1/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_246052,
*conv2d_transpose_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_246262"
 conv2d_3/StatefulPartitionedCall?
IdentityIdentity)conv2d_3/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
2__inference_conv2d_transpose_2_layer_call_fn_24613

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_246052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
??
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_25458

inputs=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOpf
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_2/stack?
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1?
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulv
conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul_1/y?
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mul_1v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/mul:z:0conv2d_transpose/mul_1:z:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_3/stack?
(conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_1?
(conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_2?
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_3?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose/BiasAdd?
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose/Relu?
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stack?
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1?
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/y?
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stack?
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1?
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2?
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/Relu?
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slice?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
(conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice_2/stack?
*conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_2/stack_1?
*conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_2/stack_2?
"conv2d_transpose_2/strided_slice_2StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_2/stack:output:03conv2d_transpose_2/strided_slice_2/stack_1:output:03conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_2v
conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/mul/y?
conv2d_transpose_2/mulMul+conv2d_transpose_2/strided_slice_1:output:0!conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/mulz
conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/mul_1/y?
conv2d_transpose_2/mul_1Mul+conv2d_transpose_2/strided_slice_2:output:0#conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/mul_1z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0conv2d_transpose_2/mul:z:0conv2d_transpose_2/mul_1:z:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_3/stack?
*conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_3/stack_1?
*conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_3/stack_2?
"conv2d_transpose_2/strided_slice_3StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_3/stack:output:03conv2d_transpose_2/strided_slice_3/stack_1:output:03conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_3?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_transpose_2/BiasAdd?
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_transpose_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D%conv2d_transpose_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????002
conv2d_3/BiasAdd?
conv2d_3/SigmoidSigmoidconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????002
conv2d_3/Sigmoid?
IdentityIdentityconv2d_3/Sigmoid:y:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????002

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?
H__inference_convolutional_layer_call_and_return_conditional_losses_25146
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource8
4sequential_1_conv2d_3_conv2d_readvariableop_resource9
5sequential_1_conv2d_3_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?+sequential_1/conv2d_3/Conv2D/ReadVariableOp?4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d/BiasAdd?
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d_1/BiasAdd?
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D&sequential/conv2d_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d_2/BiasAdd?
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d_2/Relu?
#sequential_1/conv2d_transpose/ShapeShape&sequential/conv2d_2/Relu:activations:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
3sequential_1/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice_2/stack?
5sequential_1/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_2/stack_1?
5sequential_1/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_2/stack_2?
-sequential_1/conv2d_transpose/strided_slice_2StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_2/stack:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_2?
#sequential_1/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_1/conv2d_transpose/mul/y?
!sequential_1/conv2d_transpose/mulMul6sequential_1/conv2d_transpose/strided_slice_1:output:0,sequential_1/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_1/conv2d_transpose/mul?
%sequential_1/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/mul_1/y?
#sequential_1/conv2d_transpose/mul_1Mul6sequential_1/conv2d_transpose/strided_slice_2:output:0.sequential_1/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose/mul_1?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0%sequential_1/conv2d_transpose/mul:z:0'sequential_1/conv2d_transpose/mul_1:z:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_3/stack?
5sequential_1/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_3/stack_1?
5sequential_1/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_3/stack_2?
-sequential_1/conv2d_transpose/strided_slice_3StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_3/stack:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_3?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0&sequential/conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2'
%sequential_1/conv2d_transpose/BiasAdd?
"sequential_1/conv2d_transpose/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2$
"sequential_1/conv2d_transpose/Relu?
%sequential_1/conv2d_transpose_1/ShapeShape0sequential_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
5sequential_1/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice_2/stack?
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_2?
%sequential_1/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_1/mul/y?
#sequential_1/conv2d_transpose_1/mulMul8sequential_1/conv2d_transpose_1/strided_slice_1:output:0.sequential_1/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_1/mul?
'sequential_1/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/mul_1/y?
%sequential_1/conv2d_transpose_1/mul_1Mul8sequential_1/conv2d_transpose_1/strided_slice_2:output:00sequential_1/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_1/mul_1?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:0'sequential_1/conv2d_transpose_1/mul:z:0)sequential_1/conv2d_transpose_1/mul_1:z:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_3/stack?
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_3?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:00sequential_1/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
$sequential_1/conv2d_transpose_1/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2&
$sequential_1/conv2d_transpose_1/Relu?
%sequential_1/conv2d_transpose_2/ShapeShape2sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
5sequential_1/conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice_2/stack?
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_2?
%sequential_1/conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_2/mul/y?
#sequential_1/conv2d_transpose_2/mulMul8sequential_1/conv2d_transpose_2/strided_slice_1:output:0.sequential_1/conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_2/mul?
'sequential_1/conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/mul_1/y?
%sequential_1/conv2d_transpose_2/mul_1Mul8sequential_1/conv2d_transpose_2/strided_slice_2:output:00sequential_1/conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_2/mul_1?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:0'sequential_1/conv2d_transpose_2/mul:z:0)sequential_1/conv2d_transpose_2/mul_1:z:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_3/stack?
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_3?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:02sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2)
'sequential_1/conv2d_transpose_2/BiasAdd?
$sequential_1/conv2d_transpose_2/ReluRelu0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2&
$sequential_1/conv2d_transpose_2/Relu?
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+sequential_1/conv2d_3/Conv2D/ReadVariableOp?
sequential_1/conv2d_3/Conv2DConv2D2sequential_1/conv2d_transpose_2/Relu:activations:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
2
sequential_1/conv2d_3/Conv2D?
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????002
sequential_1/conv2d_3/BiasAdd?
sequential_1/conv2d_3/SigmoidSigmoid&sequential_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????002
sequential_1/conv2d_3/Sigmoid?
IdentityIdentity!sequential_1/conv2d_3/Sigmoid:y:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp5^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????002

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:! 

_user_specified_namex
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_24475

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_243582 
conv2d/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_243792"
 conv2d_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_244002"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_3_layer_call_fn_24634

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_246262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?

?
,__inference_sequential_1_layer_call_fn_25484

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_24400

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
H__inference_convolutional_layer_call_and_return_conditional_losses_24795
input_1-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5/
+sequential_1_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_7/
+sequential_1_statefulpartitionedcall_args_8
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244512$
"sequential/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5+sequential_1_statefulpartitionedcall_args_6+sequential_1_statefulpartitionedcall_args_7+sequential_1_statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_246862&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
#__inference_signature_wrapper_24902
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_243452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????002

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_24358

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
H__inference_convolutional_layer_call_and_return_conditional_losses_24838
x-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5/
+sequential_1_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_7/
+sequential_1_statefulpartitionedcall_args_8
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallx)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244752$
"sequential/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5+sequential_1_statefulpartitionedcall_args_6+sequential_1_statefulpartitionedcall_args_7+sequential_1_statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247152&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:! 

_user_specified_namex
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_24715

inputs3
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2
identity?? conv2d_3/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputs/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_245192*
(conv2d_transpose/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_245622,
*conv2d_transpose_1/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_246052,
*conv2d_transpose_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_246262"
 conv2d_3/StatefulPartitionedCall?
IdentityIdentity)conv2d_3/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_24651
input_13
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2
identity?? conv2d_3/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinput_1/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_245192*
(conv2d_transpose/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_245622,
*conv2d_transpose_1/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_246052,
*conv2d_transpose_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_246262"
 conv2d_3/StatefulPartitionedCall?
IdentityIdentity)conv2d_3/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
-__inference_convolutional_layer_call_fn_25165
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_convolutional_layer_call_and_return_conditional_losses_248382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_24435
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_243582 
conv2d/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_243792"
 conv2d_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_244002"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?

?
,__inference_sequential_1_layer_call_fn_24726
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
??
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_25357

inputs=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOpf
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_2/stack?
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1?
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulv
conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul_1/y?
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mul_1v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/mul:z:0conv2d_transpose/mul_1:z:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_3/stack?
(conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_1?
(conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_2?
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_3?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose/BiasAdd?
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose/Relu?
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stack?
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1?
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/y?
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stack?
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1?
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2?
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/Relu?
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slice?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
(conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice_2/stack?
*conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_2/stack_1?
*conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_2/stack_2?
"conv2d_transpose_2/strided_slice_2StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_2/stack:output:03conv2d_transpose_2/strided_slice_2/stack_1:output:03conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_2v
conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/mul/y?
conv2d_transpose_2/mulMul+conv2d_transpose_2/strided_slice_1:output:0!conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/mulz
conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/mul_1/y?
conv2d_transpose_2/mul_1Mul+conv2d_transpose_2/strided_slice_2:output:0#conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/mul_1z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0conv2d_transpose_2/mul:z:0conv2d_transpose_2/mul_1:z:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_3/stack?
*conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_3/stack_1?
*conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_3/stack_2?
"conv2d_transpose_2/strided_slice_3StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_3/stack:output:03conv2d_transpose_2/strided_slice_3/stack_1:output:03conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_3?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_transpose_2/BiasAdd?
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_transpose_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D%conv2d_transpose_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????002
conv2d_3/BiasAdd?
conv2d_3/SigmoidSigmoidconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????002
conv2d_3/Sigmoid?
IdentityIdentityconv2d_3/Sigmoid:y:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????002

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_2_layer_call_fn_24408

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_244002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_25256

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_24366

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_243582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_24460
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?$
?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_24562

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_24451

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_243582 
conv2d/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_243792"
 conv2d_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_244002"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_24387

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_243792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_25209

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Relu?
IdentityIdentityconv2d_2/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????00::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_25802
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate/
+assignvariableop_5_sequential_conv2d_kernel-
)assignvariableop_6_sequential_conv2d_bias1
-assignvariableop_7_sequential_conv2d_1_kernel/
+assignvariableop_8_sequential_conv2d_1_bias1
-assignvariableop_9_sequential_conv2d_2_kernel0
,assignvariableop_10_sequential_conv2d_2_bias<
8assignvariableop_11_sequential_1_conv2d_transpose_kernel:
6assignvariableop_12_sequential_1_conv2d_transpose_bias>
:assignvariableop_13_sequential_1_conv2d_transpose_1_kernel<
8assignvariableop_14_sequential_1_conv2d_transpose_1_bias>
:assignvariableop_15_sequential_1_conv2d_transpose_2_kernel<
8assignvariableop_16_sequential_1_conv2d_transpose_2_bias4
0assignvariableop_17_sequential_1_conv2d_3_kernel2
.assignvariableop_18_sequential_1_conv2d_3_bias7
3assignvariableop_19_adam_sequential_conv2d_kernel_m5
1assignvariableop_20_adam_sequential_conv2d_bias_m9
5assignvariableop_21_adam_sequential_conv2d_1_kernel_m7
3assignvariableop_22_adam_sequential_conv2d_1_bias_m9
5assignvariableop_23_adam_sequential_conv2d_2_kernel_m7
3assignvariableop_24_adam_sequential_conv2d_2_bias_mC
?assignvariableop_25_adam_sequential_1_conv2d_transpose_kernel_mA
=assignvariableop_26_adam_sequential_1_conv2d_transpose_bias_mE
Aassignvariableop_27_adam_sequential_1_conv2d_transpose_1_kernel_mC
?assignvariableop_28_adam_sequential_1_conv2d_transpose_1_bias_mE
Aassignvariableop_29_adam_sequential_1_conv2d_transpose_2_kernel_mC
?assignvariableop_30_adam_sequential_1_conv2d_transpose_2_bias_m;
7assignvariableop_31_adam_sequential_1_conv2d_3_kernel_m9
5assignvariableop_32_adam_sequential_1_conv2d_3_bias_m7
3assignvariableop_33_adam_sequential_conv2d_kernel_v5
1assignvariableop_34_adam_sequential_conv2d_bias_v9
5assignvariableop_35_adam_sequential_conv2d_1_kernel_v7
3assignvariableop_36_adam_sequential_conv2d_1_bias_v9
5assignvariableop_37_adam_sequential_conv2d_2_kernel_v7
3assignvariableop_38_adam_sequential_conv2d_2_bias_vC
?assignvariableop_39_adam_sequential_1_conv2d_transpose_kernel_vA
=assignvariableop_40_adam_sequential_1_conv2d_transpose_bias_vE
Aassignvariableop_41_adam_sequential_1_conv2d_transpose_1_kernel_vC
?assignvariableop_42_adam_sequential_1_conv2d_transpose_1_bias_vE
Aassignvariableop_43_adam_sequential_1_conv2d_transpose_2_kernel_vC
?assignvariableop_44_adam_sequential_1_conv2d_transpose_2_bias_v;
7assignvariableop_45_adam_sequential_1_conv2d_3_kernel_v9
5assignvariableop_46_adam_sequential_1_conv2d_3_bias_v
identity_48??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*?
value?B?/B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_sequential_conv2d_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp)assignvariableop_6_sequential_conv2d_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp-assignvariableop_7_sequential_conv2d_1_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_sequential_conv2d_1_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_sequential_conv2d_2_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp,assignvariableop_10_sequential_conv2d_2_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp8assignvariableop_11_sequential_1_conv2d_transpose_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp6assignvariableop_12_sequential_1_conv2d_transpose_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp:assignvariableop_13_sequential_1_conv2d_transpose_1_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp8assignvariableop_14_sequential_1_conv2d_transpose_1_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp:assignvariableop_15_sequential_1_conv2d_transpose_2_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp8assignvariableop_16_sequential_1_conv2d_transpose_2_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp0assignvariableop_17_sequential_1_conv2d_3_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_sequential_1_conv2d_3_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_sequential_conv2d_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_sequential_conv2d_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_sequential_conv2d_1_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adam_sequential_conv2d_1_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_sequential_conv2d_2_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_sequential_conv2d_2_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp?assignvariableop_25_adam_sequential_1_conv2d_transpose_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp=assignvariableop_26_adam_sequential_1_conv2d_transpose_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpAassignvariableop_27_adam_sequential_1_conv2d_transpose_1_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp?assignvariableop_28_adam_sequential_1_conv2d_transpose_1_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpAassignvariableop_29_adam_sequential_1_conv2d_transpose_2_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp?assignvariableop_30_adam_sequential_1_conv2d_transpose_2_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adam_sequential_1_conv2d_3_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_sequential_1_conv2d_3_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_sequential_conv2d_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp1assignvariableop_34_adam_sequential_conv2d_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_sequential_conv2d_1_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_sequential_conv2d_1_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_sequential_conv2d_2_kernel_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_sequential_conv2d_2_bias_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp?assignvariableop_39_adam_sequential_1_conv2d_transpose_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp=assignvariableop_40_adam_sequential_1_conv2d_transpose_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpAassignvariableop_41_adam_sequential_1_conv2d_transpose_1_kernel_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp?assignvariableop_42_adam_sequential_1_conv2d_transpose_1_bias_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpAassignvariableop_43_adam_sequential_1_conv2d_transpose_2_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp?assignvariableop_44_adam_sequential_1_conv2d_transpose_2_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_sequential_1_conv2d_3_kernel_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_sequential_1_conv2d_3_bias_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_47?
Identity_48IdentityIdentity_47:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_48"#
identity_48Identity_48:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
?
2__inference_conv2d_transpose_1_layer_call_fn_24570

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_245622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
0__inference_conv2d_transpose_layer_call_fn_24527

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_245192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_24626

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?
H__inference_convolutional_layer_call_and_return_conditional_losses_25024
x4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource8
4sequential_1_conv2d_3_conv2d_readvariableop_resource9
5sequential_1_conv2d_3_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?+sequential_1/conv2d_3/Conv2D/ReadVariableOp?4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d/BiasAdd?
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d_1/BiasAdd?
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2DConv2D&sequential/conv2d_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d_2/BiasAdd?
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d_2/Relu?
#sequential_1/conv2d_transpose/ShapeShape&sequential/conv2d_2/Relu:activations:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape?
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stack?
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1?
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_slice?
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice_1/stack?
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1?
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1?
3sequential_1/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice_2/stack?
5sequential_1/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_2/stack_1?
5sequential_1/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_2/stack_2?
-sequential_1/conv2d_transpose/strided_slice_2StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_2/stack:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_2?
#sequential_1/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_1/conv2d_transpose/mul/y?
!sequential_1/conv2d_transpose/mulMul6sequential_1/conv2d_transpose/strided_slice_1:output:0,sequential_1/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_1/conv2d_transpose/mul?
%sequential_1/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/mul_1/y?
#sequential_1/conv2d_transpose/mul_1Mul6sequential_1/conv2d_transpose/strided_slice_2:output:0.sequential_1/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose/mul_1?
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/stack/3?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0%sequential_1/conv2d_transpose/mul:z:0'sequential_1/conv2d_transpose/mul_1:z:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stack?
3sequential_1/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_3/stack?
5sequential_1/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_3/stack_1?
5sequential_1/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_3/stack_2?
-sequential_1/conv2d_transpose/strided_slice_3StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_3/stack:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_3?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0&sequential/conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
20
.sequential_1/conv2d_transpose/conv2d_transpose?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2'
%sequential_1/conv2d_transpose/BiasAdd?
"sequential_1/conv2d_transpose/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2$
"sequential_1/conv2d_transpose/Relu?
%sequential_1/conv2d_transpose_1/ShapeShape0sequential_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shape?
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stack?
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1?
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_slice?
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice_1/stack?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1?
5sequential_1/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice_2/stack?
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_2?
%sequential_1/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_1/mul/y?
#sequential_1/conv2d_transpose_1/mulMul8sequential_1/conv2d_transpose_1/strided_slice_1:output:0.sequential_1/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_1/mul?
'sequential_1/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/mul_1/y?
%sequential_1/conv2d_transpose_1/mul_1Mul8sequential_1/conv2d_transpose_1/strided_slice_2:output:00sequential_1/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_1/mul_1?
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/stack/3?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:0'sequential_1/conv2d_transpose_1/mul:z:0)sequential_1/conv2d_transpose_1/mul_1:z:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stack?
5sequential_1/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_3/stack?
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_1?
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_2?
/sequential_1/conv2d_transpose_1/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_3?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:00sequential_1/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transpose?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_1/conv2d_transpose_1/BiasAdd?
$sequential_1/conv2d_transpose_1/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2&
$sequential_1/conv2d_transpose_1/Relu?
%sequential_1/conv2d_transpose_2/ShapeShape2sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shape?
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stack?
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1?
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_slice?
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice_1/stack?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1?
5sequential_1/conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice_2/stack?
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_2?
%sequential_1/conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_2/mul/y?
#sequential_1/conv2d_transpose_2/mulMul8sequential_1/conv2d_transpose_2/strided_slice_1:output:0.sequential_1/conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_2/mul?
'sequential_1/conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/mul_1/y?
%sequential_1/conv2d_transpose_2/mul_1Mul8sequential_1/conv2d_transpose_2/strided_slice_2:output:00sequential_1/conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_2/mul_1?
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_1/conv2d_transpose_2/stack/3?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:0'sequential_1/conv2d_transpose_2/mul:z:0)sequential_1/conv2d_transpose_2/mul_1:z:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stack?
5sequential_1/conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_3/stack?
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_1?
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_2?
/sequential_1/conv2d_transpose_2/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_3?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:02sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transpose?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2)
'sequential_1/conv2d_transpose_2/BiasAdd?
$sequential_1/conv2d_transpose_2/ReluRelu0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2&
$sequential_1/conv2d_transpose_2/Relu?
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+sequential_1/conv2d_3/Conv2D/ReadVariableOp?
sequential_1/conv2d_3/Conv2DConv2D2sequential_1/conv2d_transpose_2/Relu:activations:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
2
sequential_1/conv2d_3/Conv2D?
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????002
sequential_1/conv2d_3/BiasAdd?
sequential_1/conv2d_3/SigmoidSigmoid&sequential_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????002
sequential_1/conv2d_3/Sigmoid?
IdentityIdentity!sequential_1/conv2d_3/Sigmoid:y:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp5^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????002

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????00::::::::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:! 

_user_specified_namex"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????00D
output_18
StatefulPartitionedCall:0?????????00tensorflow/serving/predict:??
?
encoder
decoder
	optimizer
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
_training_endpoints
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?
_tf_keras_model?{"class_name": "convolutional", "name": "convolutional", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "convolutional"}, "training_config": {"loss": "mae", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

layer-0
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?*
layer-0
layer-1
layer-2
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?(
_tf_keras_sequential?({"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?
iter

beta_1

beta_2
	decay
learning_ratempmq mr!ms"mt#mu$mv%mw&mx'my(mz)m{*m|+m}v~v v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?"
	optimizer
 "
trackable_list_wrapper
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13"
trackable_list_wrapper
?
,layer_regularization_losses
-metrics
.non_trainable_variables
	variables
regularization_losses

/layers
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

kernel
bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
?

 kernel
!bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
?

"kernel
#bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
J
0
1
 2
!3
"4
#5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
 2
!3
"4
#5"
trackable_list_wrapper
?
<layer_regularization_losses
=metrics
>non_trainable_variables
	variables
regularization_losses

?layers
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

$kernel
%bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
?

&kernel
'bias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
?

(kernel
)bias
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
?

*kernel
+bias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
X
$0
%1
&2
'3
(4
)5
*6
+7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
$0
%1
&2
'3
(4
)5
*6
+7"
trackable_list_wrapper
?
Player_regularization_losses
Qmetrics
Rnon_trainable_variables
	variables
regularization_losses

Slayers
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
2:0@2sequential/conv2d/kernel
$:"@2sequential/conv2d/bias
4:2@ 2sequential/conv2d_1/kernel
&:$ 2sequential/conv2d_1/bias
4:2 2sequential/conv2d_2/kernel
&:$2sequential/conv2d_2/bias
>:<2$sequential_1/conv2d_transpose/kernel
0:.2"sequential_1/conv2d_transpose/bias
@:> 2&sequential_1/conv2d_transpose_1/kernel
2:0 2$sequential_1/conv2d_transpose_1/bias
@:>@ 2&sequential_1/conv2d_transpose_2/kernel
2:0@2$sequential_1/conv2d_transpose_2/bias
6:4@2sequential_1/conv2d_3/kernel
(:&2sequential_1/conv2d_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Tlayer_regularization_losses
Unon_trainable_variables
0	variables
Vmetrics
1regularization_losses

Wlayers
2trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
Xlayer_regularization_losses
Ynon_trainable_variables
4	variables
Zmetrics
5regularization_losses

[layers
6trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
\layer_regularization_losses
]non_trainable_variables
8	variables
^metrics
9regularization_losses

_layers
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5

0
1
2"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
`layer_regularization_losses
anon_trainable_variables
@	variables
bmetrics
Aregularization_losses

clayers
Btrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
dlayer_regularization_losses
enon_trainable_variables
D	variables
fmetrics
Eregularization_losses

glayers
Ftrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
hlayer_regularization_losses
inon_trainable_variables
H	variables
jmetrics
Iregularization_losses

klayers
Jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
llayer_regularization_losses
mnon_trainable_variables
L	variables
nmetrics
Mregularization_losses

olayers
Ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
7:5@2Adam/sequential/conv2d/kernel/m
):'@2Adam/sequential/conv2d/bias/m
9:7@ 2!Adam/sequential/conv2d_1/kernel/m
+:) 2Adam/sequential/conv2d_1/bias/m
9:7 2!Adam/sequential/conv2d_2/kernel/m
+:)2Adam/sequential/conv2d_2/bias/m
C:A2+Adam/sequential_1/conv2d_transpose/kernel/m
5:32)Adam/sequential_1/conv2d_transpose/bias/m
E:C 2-Adam/sequential_1/conv2d_transpose_1/kernel/m
7:5 2+Adam/sequential_1/conv2d_transpose_1/bias/m
E:C@ 2-Adam/sequential_1/conv2d_transpose_2/kernel/m
7:5@2+Adam/sequential_1/conv2d_transpose_2/bias/m
;:9@2#Adam/sequential_1/conv2d_3/kernel/m
-:+2!Adam/sequential_1/conv2d_3/bias/m
7:5@2Adam/sequential/conv2d/kernel/v
):'@2Adam/sequential/conv2d/bias/v
9:7@ 2!Adam/sequential/conv2d_1/kernel/v
+:) 2Adam/sequential/conv2d_1/bias/v
9:7 2!Adam/sequential/conv2d_2/kernel/v
+:)2Adam/sequential/conv2d_2/bias/v
C:A2+Adam/sequential_1/conv2d_transpose/kernel/v
5:32)Adam/sequential_1/conv2d_transpose/bias/v
E:C 2-Adam/sequential_1/conv2d_transpose_1/kernel/v
7:5 2+Adam/sequential_1/conv2d_transpose_1/bias/v
E:C@ 2-Adam/sequential_1/conv2d_transpose_2/kernel/v
7:5@2+Adam/sequential_1/conv2d_transpose_2/bias/v
;:9@2#Adam/sequential_1/conv2d_3/kernel/v
-:+2!Adam/sequential_1/conv2d_3/bias/v
?2?
-__inference_convolutional_layer_call_fn_24874
-__inference_convolutional_layer_call_fn_25184
-__inference_convolutional_layer_call_fn_25165
-__inference_convolutional_layer_call_fn_24855?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_convolutional_layer_call_and_return_conditional_losses_25146
H__inference_convolutional_layer_call_and_return_conditional_losses_25024
H__inference_convolutional_layer_call_and_return_conditional_losses_24795
H__inference_convolutional_layer_call_and_return_conditional_losses_24815?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
 __inference__wrapped_model_24345?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????00
?2?
*__inference_sequential_layer_call_fn_25245
*__inference_sequential_layer_call_fn_25256
*__inference_sequential_layer_call_fn_24484
*__inference_sequential_layer_call_fn_24460?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_24422
E__inference_sequential_layer_call_and_return_conditional_losses_25209
E__inference_sequential_layer_call_and_return_conditional_losses_25234
E__inference_sequential_layer_call_and_return_conditional_losses_24435?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_1_layer_call_fn_24697
,__inference_sequential_1_layer_call_fn_25471
,__inference_sequential_1_layer_call_fn_25484
,__inference_sequential_1_layer_call_fn_24726?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_24667
G__inference_sequential_1_layer_call_and_return_conditional_losses_25458
G__inference_sequential_1_layer_call_and_return_conditional_losses_25357
G__inference_sequential_1_layer_call_and_return_conditional_losses_24651?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
2B0
#__inference_signature_wrapper_24902input_1
?2?
&__inference_conv2d_layer_call_fn_24366?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_24358?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
(__inference_conv2d_1_layer_call_fn_24387?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24379?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
(__inference_conv2d_2_layer_call_fn_24408?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_24400?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
0__inference_conv2d_transpose_layer_call_fn_24527?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_24519?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
2__inference_conv2d_transpose_1_layer_call_fn_24570?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_24562?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
2__inference_conv2d_transpose_2_layer_call_fn_24613?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_24605?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
(__inference_conv2d_3_layer_call_fn_24634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_24626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@?
 __inference__wrapped_model_24345? !"#$%&'()*+8?5
.?+
)?&
input_1?????????00
? ";?8
6
output_1*?'
output_1?????????00?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24379? !I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
(__inference_conv2d_1_layer_call_fn_24387? !I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_24400?"#I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
(__inference_conv2d_2_layer_call_fn_24408?"#I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
C__inference_conv2d_3_layer_call_and_return_conditional_losses_24626?*+I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????
? ?
(__inference_conv2d_3_layer_call_fn_24634?*+I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+????????????????????????????
A__inference_conv2d_layer_call_and_return_conditional_losses_24358?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
&__inference_conv2d_layer_call_fn_24366?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????@?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_24562?&'I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
2__inference_conv2d_transpose_1_layer_call_fn_24570?&'I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_24605?()I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
2__inference_conv2d_transpose_2_layer_call_fn_24613?()I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_24519?$%I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
0__inference_conv2d_transpose_layer_call_fn_24527?$%I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
H__inference_convolutional_layer_call_and_return_conditional_losses_24795? !"#$%&'()*+<?9
2?/
)?&
input_1?????????00
p
? "??<
5?2
0+???????????????????????????
? ?
H__inference_convolutional_layer_call_and_return_conditional_losses_24815? !"#$%&'()*+<?9
2?/
)?&
input_1?????????00
p 
? "??<
5?2
0+???????????????????????????
? ?
H__inference_convolutional_layer_call_and_return_conditional_losses_25024w !"#$%&'()*+6?3
,?)
#? 
x?????????00
p
? "-?*
#? 
0?????????00
? ?
H__inference_convolutional_layer_call_and_return_conditional_losses_25146w !"#$%&'()*+6?3
,?)
#? 
x?????????00
p 
? "-?*
#? 
0?????????00
? ?
-__inference_convolutional_layer_call_fn_24855? !"#$%&'()*+<?9
2?/
)?&
input_1?????????00
p
? "2?/+????????????????????????????
-__inference_convolutional_layer_call_fn_24874? !"#$%&'()*+<?9
2?/
)?&
input_1?????????00
p 
? "2?/+????????????????????????????
-__inference_convolutional_layer_call_fn_25165| !"#$%&'()*+6?3
,?)
#? 
x?????????00
p
? "2?/+????????????????????????????
-__inference_convolutional_layer_call_fn_25184| !"#$%&'()*+6?3
,?)
#? 
x?????????00
p 
? "2?/+????????????????????????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_24651?$%&'()*+@?=
6?3
)?&
input_1?????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_24667?$%&'()*+@?=
6?3
)?&
input_1?????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_25357z$%&'()*+??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????00
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_25458z$%&'()*+??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????00
? ?
,__inference_sequential_1_layer_call_fn_24697?$%&'()*+@?=
6?3
)?&
input_1?????????
p

 
? "2?/+????????????????????????????
,__inference_sequential_1_layer_call_fn_24726?$%&'()*+@?=
6?3
)?&
input_1?????????
p 

 
? "2?/+????????????????????????????
,__inference_sequential_1_layer_call_fn_25471$%&'()*+??<
5?2
(?%
inputs?????????
p

 
? "2?/+????????????????????????????
,__inference_sequential_1_layer_call_fn_25484$%&'()*+??<
5?2
(?%
inputs?????????
p 

 
? "2?/+????????????????????????????
E__inference_sequential_layer_call_and_return_conditional_losses_24422y !"#@?=
6?3
)?&
input_1?????????00
p

 
? "-?*
#? 
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_24435y !"#@?=
6?3
)?&
input_1?????????00
p 

 
? "-?*
#? 
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_25209x !"#??<
5?2
(?%
inputs?????????00
p

 
? "-?*
#? 
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_25234x !"#??<
5?2
(?%
inputs?????????00
p 

 
? "-?*
#? 
0?????????
? ?
*__inference_sequential_layer_call_fn_24460l !"#@?=
6?3
)?&
input_1?????????00
p

 
? " ???????????
*__inference_sequential_layer_call_fn_24484l !"#@?=
6?3
)?&
input_1?????????00
p 

 
? " ???????????
*__inference_sequential_layer_call_fn_25245k !"#??<
5?2
(?%
inputs?????????00
p

 
? " ???????????
*__inference_sequential_layer_call_fn_25256k !"#??<
5?2
(?%
inputs?????????00
p 

 
? " ???????????
#__inference_signature_wrapper_24902? !"#$%&'()*+C?@
? 
9?6
4
input_1)?&
input_1?????????00";?8
6
output_1*?'
output_1?????????00