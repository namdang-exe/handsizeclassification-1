€ђ
Ђэ
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
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8¬Є
f
	Adam/iterVarHandleOp*
shape: *
shared_name	Adam/iter*
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shape: *
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
shape: *
shared_nameAdam/beta_2*
dtype0*
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shape: *
shared_name
Adam/decay*
dtype0*
_output_shapes
: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
§
 model/encode_block/conv/VariableVarHandleOp*
shape:@*1
shared_name" model/encode_block/conv/Variable*
dtype0*
_output_shapes
: 
Э
4model/encode_block/conv/Variable/Read/ReadVariableOpReadVariableOp model/encode_block/conv/Variable*
dtype0*&
_output_shapes
:@
Ь
"model/encode_block/conv/Variable_1VarHandleOp*
shape:@*3
shared_name$"model/encode_block/conv/Variable_1*
dtype0*
_output_shapes
: 
Х
6model/encode_block/conv/Variable_1/Read/ReadVariableOpReadVariableOp"model/encode_block/conv/Variable_1*
dtype0*
_output_shapes
:@
ђ
$model/encode_block_1/conv_1/VariableVarHandleOp*
shape:@@*5
shared_name&$model/encode_block_1/conv_1/Variable*
dtype0*
_output_shapes
: 
•
8model/encode_block_1/conv_1/Variable/Read/ReadVariableOpReadVariableOp$model/encode_block_1/conv_1/Variable*
dtype0*&
_output_shapes
:@@
§
&model/encode_block_1/conv_1/Variable_1VarHandleOp*
shape:@*7
shared_name(&model/encode_block_1/conv_1/Variable_1*
dtype0*
_output_shapes
: 
Э
:model/encode_block_1/conv_1/Variable_1/Read/ReadVariableOpReadVariableOp&model/encode_block_1/conv_1/Variable_1*
dtype0*
_output_shapes
:@
ђ
$model/encode_block_2/conv_2/VariableVarHandleOp*
shape:@@*5
shared_name&$model/encode_block_2/conv_2/Variable*
dtype0*
_output_shapes
: 
•
8model/encode_block_2/conv_2/Variable/Read/ReadVariableOpReadVariableOp$model/encode_block_2/conv_2/Variable*
dtype0*&
_output_shapes
:@@
§
&model/encode_block_2/conv_2/Variable_1VarHandleOp*
shape:@*7
shared_name(&model/encode_block_2/conv_2/Variable_1*
dtype0*
_output_shapes
: 
Э
:model/encode_block_2/conv_2/Variable_1/Read/ReadVariableOpReadVariableOp&model/encode_block_2/conv_2/Variable_1*
dtype0*
_output_shapes
:@
ђ
$model/encode_block_3/conv_3/VariableVarHandleOp*
shape:@@*5
shared_name&$model/encode_block_3/conv_3/Variable*
dtype0*
_output_shapes
: 
•
8model/encode_block_3/conv_3/Variable/Read/ReadVariableOpReadVariableOp$model/encode_block_3/conv_3/Variable*
dtype0*&
_output_shapes
:@@
§
&model/encode_block_3/conv_3/Variable_1VarHandleOp*
shape:@*7
shared_name(&model/encode_block_3/conv_3/Variable_1*
dtype0*
_output_shapes
: 
Э
:model/encode_block_3/conv_3/Variable_1/Read/ReadVariableOpReadVariableOp&model/encode_block_3/conv_3/Variable_1*
dtype0*
_output_shapes
:@
®
"model/decode_block/conv_4/VariableVarHandleOp*
shape:@ *3
shared_name$"model/decode_block/conv_4/Variable*
dtype0*
_output_shapes
: 
°
6model/decode_block/conv_4/Variable/Read/ReadVariableOpReadVariableOp"model/decode_block/conv_4/Variable*
dtype0*&
_output_shapes
:@ 
†
$model/decode_block/conv_4/Variable_1VarHandleOp*
shape: *5
shared_name&$model/decode_block/conv_4/Variable_1*
dtype0*
_output_shapes
: 
Щ
8model/decode_block/conv_4/Variable_1/Read/ReadVariableOpReadVariableOp$model/decode_block/conv_4/Variable_1*
dtype0*
_output_shapes
: 
ђ
$model/decode_block_1/conv_5/VariableVarHandleOp*
shape:  *5
shared_name&$model/decode_block_1/conv_5/Variable*
dtype0*
_output_shapes
: 
•
8model/decode_block_1/conv_5/Variable/Read/ReadVariableOpReadVariableOp$model/decode_block_1/conv_5/Variable*
dtype0*&
_output_shapes
:  
§
&model/decode_block_1/conv_5/Variable_1VarHandleOp*
shape: *7
shared_name(&model/decode_block_1/conv_5/Variable_1*
dtype0*
_output_shapes
: 
Э
:model/decode_block_1/conv_5/Variable_1/Read/ReadVariableOpReadVariableOp&model/decode_block_1/conv_5/Variable_1*
dtype0*
_output_shapes
: 
ђ
$model/decode_block_2/conv_6/VariableVarHandleOp*
shape:@ *5
shared_name&$model/decode_block_2/conv_6/Variable*
dtype0*
_output_shapes
: 
•
8model/decode_block_2/conv_6/Variable/Read/ReadVariableOpReadVariableOp$model/decode_block_2/conv_6/Variable*
dtype0*&
_output_shapes
:@ 
§
&model/decode_block_2/conv_6/Variable_1VarHandleOp*
shape: *7
shared_name(&model/decode_block_2/conv_6/Variable_1*
dtype0*
_output_shapes
: 
Э
:model/decode_block_2/conv_6/Variable_1/Read/ReadVariableOpReadVariableOp&model/decode_block_2/conv_6/Variable_1*
dtype0*
_output_shapes
: 
ђ
$model/decode_block_3/conv_7/VariableVarHandleOp*
shape:@ *5
shared_name&$model/decode_block_3/conv_7/Variable*
dtype0*
_output_shapes
: 
•
8model/decode_block_3/conv_7/Variable/Read/ReadVariableOpReadVariableOp$model/decode_block_3/conv_7/Variable*
dtype0*&
_output_shapes
:@ 
§
&model/decode_block_3/conv_7/Variable_1VarHandleOp*
shape: *7
shared_name(&model/decode_block_3/conv_7/Variable_1*
dtype0*
_output_shapes
: 
Э
:model/decode_block_3/conv_7/Variable_1/Read/ReadVariableOpReadVariableOp&model/decode_block_3/conv_7/Variable_1*
dtype0*
_output_shapes
: 
ђ
$model/decode_block_4/conv_8/VariableVarHandleOp*
shape:`*5
shared_name&$model/decode_block_4/conv_8/Variable*
dtype0*
_output_shapes
: 
•
8model/decode_block_4/conv_8/Variable/Read/ReadVariableOpReadVariableOp$model/decode_block_4/conv_8/Variable*
dtype0*&
_output_shapes
:`
§
&model/decode_block_4/conv_8/Variable_1VarHandleOp*
shape:*7
shared_name(&model/decode_block_4/conv_8/Variable_1*
dtype0*
_output_shapes
: 
Э
:model/decode_block_4/conv_8/Variable_1/Read/ReadVariableOpReadVariableOp&model/decode_block_4/conv_8/Variable_1*
dtype0*
_output_shapes
:
≤
'Adam/model/encode_block/conv/Variable/mVarHandleOp*
shape:@*8
shared_name)'Adam/model/encode_block/conv/Variable/m*
dtype0*
_output_shapes
: 
Ђ
;Adam/model/encode_block/conv/Variable/m/Read/ReadVariableOpReadVariableOp'Adam/model/encode_block/conv/Variable/m*
dtype0*&
_output_shapes
:@
™
)Adam/model/encode_block/conv/Variable/m_1VarHandleOp*
shape:@*:
shared_name+)Adam/model/encode_block/conv/Variable/m_1*
dtype0*
_output_shapes
: 
£
=Adam/model/encode_block/conv/Variable/m_1/Read/ReadVariableOpReadVariableOp)Adam/model/encode_block/conv/Variable/m_1*
dtype0*
_output_shapes
:@
Ї
+Adam/model/encode_block_1/conv_1/Variable/mVarHandleOp*
shape:@@*<
shared_name-+Adam/model/encode_block_1/conv_1/Variable/m*
dtype0*
_output_shapes
: 
≥
?Adam/model/encode_block_1/conv_1/Variable/m/Read/ReadVariableOpReadVariableOp+Adam/model/encode_block_1/conv_1/Variable/m*
dtype0*&
_output_shapes
:@@
≤
-Adam/model/encode_block_1/conv_1/Variable/m_1VarHandleOp*
shape:@*>
shared_name/-Adam/model/encode_block_1/conv_1/Variable/m_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/encode_block_1/conv_1/Variable/m_1/Read/ReadVariableOpReadVariableOp-Adam/model/encode_block_1/conv_1/Variable/m_1*
dtype0*
_output_shapes
:@
Ї
+Adam/model/encode_block_2/conv_2/Variable/mVarHandleOp*
shape:@@*<
shared_name-+Adam/model/encode_block_2/conv_2/Variable/m*
dtype0*
_output_shapes
: 
≥
?Adam/model/encode_block_2/conv_2/Variable/m/Read/ReadVariableOpReadVariableOp+Adam/model/encode_block_2/conv_2/Variable/m*
dtype0*&
_output_shapes
:@@
≤
-Adam/model/encode_block_2/conv_2/Variable/m_1VarHandleOp*
shape:@*>
shared_name/-Adam/model/encode_block_2/conv_2/Variable/m_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/encode_block_2/conv_2/Variable/m_1/Read/ReadVariableOpReadVariableOp-Adam/model/encode_block_2/conv_2/Variable/m_1*
dtype0*
_output_shapes
:@
Ї
+Adam/model/encode_block_3/conv_3/Variable/mVarHandleOp*
shape:@@*<
shared_name-+Adam/model/encode_block_3/conv_3/Variable/m*
dtype0*
_output_shapes
: 
≥
?Adam/model/encode_block_3/conv_3/Variable/m/Read/ReadVariableOpReadVariableOp+Adam/model/encode_block_3/conv_3/Variable/m*
dtype0*&
_output_shapes
:@@
≤
-Adam/model/encode_block_3/conv_3/Variable/m_1VarHandleOp*
shape:@*>
shared_name/-Adam/model/encode_block_3/conv_3/Variable/m_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/encode_block_3/conv_3/Variable/m_1/Read/ReadVariableOpReadVariableOp-Adam/model/encode_block_3/conv_3/Variable/m_1*
dtype0*
_output_shapes
:@
ґ
)Adam/model/decode_block/conv_4/Variable/mVarHandleOp*
shape:@ *:
shared_name+)Adam/model/decode_block/conv_4/Variable/m*
dtype0*
_output_shapes
: 
ѓ
=Adam/model/decode_block/conv_4/Variable/m/Read/ReadVariableOpReadVariableOp)Adam/model/decode_block/conv_4/Variable/m*
dtype0*&
_output_shapes
:@ 
Ѓ
+Adam/model/decode_block/conv_4/Variable/m_1VarHandleOp*
shape: *<
shared_name-+Adam/model/decode_block/conv_4/Variable/m_1*
dtype0*
_output_shapes
: 
І
?Adam/model/decode_block/conv_4/Variable/m_1/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block/conv_4/Variable/m_1*
dtype0*
_output_shapes
: 
Ї
+Adam/model/decode_block_1/conv_5/Variable/mVarHandleOp*
shape:  *<
shared_name-+Adam/model/decode_block_1/conv_5/Variable/m*
dtype0*
_output_shapes
: 
≥
?Adam/model/decode_block_1/conv_5/Variable/m/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block_1/conv_5/Variable/m*
dtype0*&
_output_shapes
:  
≤
-Adam/model/decode_block_1/conv_5/Variable/m_1VarHandleOp*
shape: *>
shared_name/-Adam/model/decode_block_1/conv_5/Variable/m_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/decode_block_1/conv_5/Variable/m_1/Read/ReadVariableOpReadVariableOp-Adam/model/decode_block_1/conv_5/Variable/m_1*
dtype0*
_output_shapes
: 
Ї
+Adam/model/decode_block_2/conv_6/Variable/mVarHandleOp*
shape:@ *<
shared_name-+Adam/model/decode_block_2/conv_6/Variable/m*
dtype0*
_output_shapes
: 
≥
?Adam/model/decode_block_2/conv_6/Variable/m/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block_2/conv_6/Variable/m*
dtype0*&
_output_shapes
:@ 
≤
-Adam/model/decode_block_2/conv_6/Variable/m_1VarHandleOp*
shape: *>
shared_name/-Adam/model/decode_block_2/conv_6/Variable/m_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/decode_block_2/conv_6/Variable/m_1/Read/ReadVariableOpReadVariableOp-Adam/model/decode_block_2/conv_6/Variable/m_1*
dtype0*
_output_shapes
: 
Ї
+Adam/model/decode_block_3/conv_7/Variable/mVarHandleOp*
shape:@ *<
shared_name-+Adam/model/decode_block_3/conv_7/Variable/m*
dtype0*
_output_shapes
: 
≥
?Adam/model/decode_block_3/conv_7/Variable/m/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block_3/conv_7/Variable/m*
dtype0*&
_output_shapes
:@ 
≤
-Adam/model/decode_block_3/conv_7/Variable/m_1VarHandleOp*
shape: *>
shared_name/-Adam/model/decode_block_3/conv_7/Variable/m_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/decode_block_3/conv_7/Variable/m_1/Read/ReadVariableOpReadVariableOp-Adam/model/decode_block_3/conv_7/Variable/m_1*
dtype0*
_output_shapes
: 
Ї
+Adam/model/decode_block_4/conv_8/Variable/mVarHandleOp*
shape:`*<
shared_name-+Adam/model/decode_block_4/conv_8/Variable/m*
dtype0*
_output_shapes
: 
≥
?Adam/model/decode_block_4/conv_8/Variable/m/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block_4/conv_8/Variable/m*
dtype0*&
_output_shapes
:`
≤
-Adam/model/decode_block_4/conv_8/Variable/m_1VarHandleOp*
shape:*>
shared_name/-Adam/model/decode_block_4/conv_8/Variable/m_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/decode_block_4/conv_8/Variable/m_1/Read/ReadVariableOpReadVariableOp-Adam/model/decode_block_4/conv_8/Variable/m_1*
dtype0*
_output_shapes
:
≤
'Adam/model/encode_block/conv/Variable/vVarHandleOp*
shape:@*8
shared_name)'Adam/model/encode_block/conv/Variable/v*
dtype0*
_output_shapes
: 
Ђ
;Adam/model/encode_block/conv/Variable/v/Read/ReadVariableOpReadVariableOp'Adam/model/encode_block/conv/Variable/v*
dtype0*&
_output_shapes
:@
™
)Adam/model/encode_block/conv/Variable/v_1VarHandleOp*
shape:@*:
shared_name+)Adam/model/encode_block/conv/Variable/v_1*
dtype0*
_output_shapes
: 
£
=Adam/model/encode_block/conv/Variable/v_1/Read/ReadVariableOpReadVariableOp)Adam/model/encode_block/conv/Variable/v_1*
dtype0*
_output_shapes
:@
Ї
+Adam/model/encode_block_1/conv_1/Variable/vVarHandleOp*
shape:@@*<
shared_name-+Adam/model/encode_block_1/conv_1/Variable/v*
dtype0*
_output_shapes
: 
≥
?Adam/model/encode_block_1/conv_1/Variable/v/Read/ReadVariableOpReadVariableOp+Adam/model/encode_block_1/conv_1/Variable/v*
dtype0*&
_output_shapes
:@@
≤
-Adam/model/encode_block_1/conv_1/Variable/v_1VarHandleOp*
shape:@*>
shared_name/-Adam/model/encode_block_1/conv_1/Variable/v_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/encode_block_1/conv_1/Variable/v_1/Read/ReadVariableOpReadVariableOp-Adam/model/encode_block_1/conv_1/Variable/v_1*
dtype0*
_output_shapes
:@
Ї
+Adam/model/encode_block_2/conv_2/Variable/vVarHandleOp*
shape:@@*<
shared_name-+Adam/model/encode_block_2/conv_2/Variable/v*
dtype0*
_output_shapes
: 
≥
?Adam/model/encode_block_2/conv_2/Variable/v/Read/ReadVariableOpReadVariableOp+Adam/model/encode_block_2/conv_2/Variable/v*
dtype0*&
_output_shapes
:@@
≤
-Adam/model/encode_block_2/conv_2/Variable/v_1VarHandleOp*
shape:@*>
shared_name/-Adam/model/encode_block_2/conv_2/Variable/v_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/encode_block_2/conv_2/Variable/v_1/Read/ReadVariableOpReadVariableOp-Adam/model/encode_block_2/conv_2/Variable/v_1*
dtype0*
_output_shapes
:@
Ї
+Adam/model/encode_block_3/conv_3/Variable/vVarHandleOp*
shape:@@*<
shared_name-+Adam/model/encode_block_3/conv_3/Variable/v*
dtype0*
_output_shapes
: 
≥
?Adam/model/encode_block_3/conv_3/Variable/v/Read/ReadVariableOpReadVariableOp+Adam/model/encode_block_3/conv_3/Variable/v*
dtype0*&
_output_shapes
:@@
≤
-Adam/model/encode_block_3/conv_3/Variable/v_1VarHandleOp*
shape:@*>
shared_name/-Adam/model/encode_block_3/conv_3/Variable/v_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/encode_block_3/conv_3/Variable/v_1/Read/ReadVariableOpReadVariableOp-Adam/model/encode_block_3/conv_3/Variable/v_1*
dtype0*
_output_shapes
:@
ґ
)Adam/model/decode_block/conv_4/Variable/vVarHandleOp*
shape:@ *:
shared_name+)Adam/model/decode_block/conv_4/Variable/v*
dtype0*
_output_shapes
: 
ѓ
=Adam/model/decode_block/conv_4/Variable/v/Read/ReadVariableOpReadVariableOp)Adam/model/decode_block/conv_4/Variable/v*
dtype0*&
_output_shapes
:@ 
Ѓ
+Adam/model/decode_block/conv_4/Variable/v_1VarHandleOp*
shape: *<
shared_name-+Adam/model/decode_block/conv_4/Variable/v_1*
dtype0*
_output_shapes
: 
І
?Adam/model/decode_block/conv_4/Variable/v_1/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block/conv_4/Variable/v_1*
dtype0*
_output_shapes
: 
Ї
+Adam/model/decode_block_1/conv_5/Variable/vVarHandleOp*
shape:  *<
shared_name-+Adam/model/decode_block_1/conv_5/Variable/v*
dtype0*
_output_shapes
: 
≥
?Adam/model/decode_block_1/conv_5/Variable/v/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block_1/conv_5/Variable/v*
dtype0*&
_output_shapes
:  
≤
-Adam/model/decode_block_1/conv_5/Variable/v_1VarHandleOp*
shape: *>
shared_name/-Adam/model/decode_block_1/conv_5/Variable/v_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/decode_block_1/conv_5/Variable/v_1/Read/ReadVariableOpReadVariableOp-Adam/model/decode_block_1/conv_5/Variable/v_1*
dtype0*
_output_shapes
: 
Ї
+Adam/model/decode_block_2/conv_6/Variable/vVarHandleOp*
shape:@ *<
shared_name-+Adam/model/decode_block_2/conv_6/Variable/v*
dtype0*
_output_shapes
: 
≥
?Adam/model/decode_block_2/conv_6/Variable/v/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block_2/conv_6/Variable/v*
dtype0*&
_output_shapes
:@ 
≤
-Adam/model/decode_block_2/conv_6/Variable/v_1VarHandleOp*
shape: *>
shared_name/-Adam/model/decode_block_2/conv_6/Variable/v_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/decode_block_2/conv_6/Variable/v_1/Read/ReadVariableOpReadVariableOp-Adam/model/decode_block_2/conv_6/Variable/v_1*
dtype0*
_output_shapes
: 
Ї
+Adam/model/decode_block_3/conv_7/Variable/vVarHandleOp*
shape:@ *<
shared_name-+Adam/model/decode_block_3/conv_7/Variable/v*
dtype0*
_output_shapes
: 
≥
?Adam/model/decode_block_3/conv_7/Variable/v/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block_3/conv_7/Variable/v*
dtype0*&
_output_shapes
:@ 
≤
-Adam/model/decode_block_3/conv_7/Variable/v_1VarHandleOp*
shape: *>
shared_name/-Adam/model/decode_block_3/conv_7/Variable/v_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/decode_block_3/conv_7/Variable/v_1/Read/ReadVariableOpReadVariableOp-Adam/model/decode_block_3/conv_7/Variable/v_1*
dtype0*
_output_shapes
: 
Ї
+Adam/model/decode_block_4/conv_8/Variable/vVarHandleOp*
shape:`*<
shared_name-+Adam/model/decode_block_4/conv_8/Variable/v*
dtype0*
_output_shapes
: 
≥
?Adam/model/decode_block_4/conv_8/Variable/v/Read/ReadVariableOpReadVariableOp+Adam/model/decode_block_4/conv_8/Variable/v*
dtype0*&
_output_shapes
:`
≤
-Adam/model/decode_block_4/conv_8/Variable/v_1VarHandleOp*
shape:*>
shared_name/-Adam/model/decode_block_4/conv_8/Variable/v_1*
dtype0*
_output_shapes
: 
Ђ
AAdam/model/decode_block_4/conv_8/Variable/v_1/Read/ReadVariableOpReadVariableOp-Adam/model/decode_block_4/conv_8/Variable/v_1*
dtype0*
_output_shapes
:

NoOpNoOp
ё~
ConstConst"/device:CPU:0*Щ~
valueП~BМ~ BЕ~
Џ
	enc_0
	enc_1
	enc_2
	enc_3

dec_1a

dec_1b

dec_2a

dec_3a
	dec_out

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
r
	dilations
conv_encode
	variables
trainable_variables
regularization_losses
	keras_api
r
	dilations
conv_encode
	variables
trainable_variables
regularization_losses
	keras_api
r
	dilations
conv_encode
	variables
trainable_variables
 regularization_losses
!	keras_api
r
"	dilations
#conv_encode
$	variables
%trainable_variables
&regularization_losses
'	keras_api
g
	(shape
)conv
*	variables
+trainable_variables
,regularization_losses
-	keras_api
g
	.shape
/conv
0	variables
1trainable_variables
2regularization_losses
3	keras_api
g
	4shape
5conv
6	variables
7trainable_variables
8regularization_losses
9	keras_api
g
	:shape
;conv
<	variables
=trainable_variables
>regularization_losses
?	keras_api
g
	@shape
Aconv
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
®
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_rateKmЧLmШMmЩNmЪOmЫPmЬQmЭRmЮSmЯTm†Um°VmҐWm£Xm§Ym•Zm¶[mІ\m®Kv©Lv™MvЂNvђOv≠PvЃQvѓRv∞Sv±Tv≤Uv≥VvіWvµXvґYvЈZvЄ[vє\vЇ
Ж
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12
X13
Y14
Z15
[16
\17
Ж
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12
X13
Y14
Z15
[16
\17
 
Ъ
	variables
]layer_regularization_losses

^layers
trainable_variables
_metrics
regularization_losses
`non_trainable_variables
 
 
h
adr
Kw
Lb
b	variables
ctrainable_variables
dregularization_losses
e	keras_api

K0
L1

K0
L1
 
Ъ
	variables
flayer_regularization_losses

glayers
trainable_variables
hmetrics
regularization_losses
inon_trainable_variables
 
h
jdr
Mw
Nb
k	variables
ltrainable_variables
mregularization_losses
n	keras_api

M0
N1

M0
N1
 
Ъ
	variables
olayer_regularization_losses

players
trainable_variables
qmetrics
regularization_losses
rnon_trainable_variables
 
h
sdr
Ow
Pb
t	variables
utrainable_variables
vregularization_losses
w	keras_api

O0
P1

O0
P1
 
Ъ
	variables
xlayer_regularization_losses

ylayers
trainable_variables
zmetrics
 regularization_losses
{non_trainable_variables
 
i
|dr
Qw
Rb
}	variables
~trainable_variables
regularization_losses
А	keras_api

Q0
R1

Q0
R1
 
Ю
$	variables
 Бlayer_regularization_losses
Вlayers
%trainable_variables
Гmetrics
&regularization_losses
Дnon_trainable_variables
 
m
Еdr
Sw
Tb
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api

S0
T1

S0
T1
 
Ю
*	variables
 Кlayer_regularization_losses
Лlayers
+trainable_variables
Мmetrics
,regularization_losses
Нnon_trainable_variables
 
m
Оdr
Uw
Vb
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api

U0
V1

U0
V1
 
Ю
0	variables
 Уlayer_regularization_losses
Фlayers
1trainable_variables
Хmetrics
2regularization_losses
Цnon_trainable_variables
 
m
Чdr
Ww
Xb
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api

W0
X1

W0
X1
 
Ю
6	variables
 Ьlayer_regularization_losses
Эlayers
7trainable_variables
Юmetrics
8regularization_losses
Яnon_trainable_variables
 
m
†dr
Yw
Zb
°	variables
Ґtrainable_variables
£regularization_losses
§	keras_api

Y0
Z1

Y0
Z1
 
Ю
<	variables
 •layer_regularization_losses
¶layers
=trainable_variables
Іmetrics
>regularization_losses
®non_trainable_variables
 
m
©dr
[w
\b
™	variables
Ђtrainable_variables
ђregularization_losses
≠	keras_api

[0
\1

[0
\1
 
Ю
B	variables
 Ѓlayer_regularization_losses
ѓlayers
Ctrainable_variables
∞metrics
Dregularization_losses
±non_trainable_variables
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
\Z
VARIABLE_VALUE model/encode_block/conv/Variable&variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"model/encode_block/conv/Variable_1&variables/1/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$model/encode_block_1/conv_1/Variable&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&model/encode_block_1/conv_1/Variable_1&variables/3/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$model/encode_block_2/conv_2/Variable&variables/4/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&model/encode_block_2/conv_2/Variable_1&variables/5/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$model/encode_block_3/conv_3/Variable&variables/6/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&model/encode_block_3/conv_3/Variable_1&variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"model/decode_block/conv_4/Variable&variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$model/decode_block/conv_4/Variable_1&variables/9/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$model/decode_block_1/conv_5/Variable'variables/10/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&model/decode_block_1/conv_5/Variable_1'variables/11/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$model/decode_block_2/conv_6/Variable'variables/12/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&model/decode_block_2/conv_6/Variable_1'variables/13/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$model/decode_block_3/conv_7/Variable'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&model/decode_block_3/conv_7/Variable_1'variables/15/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$model/decode_block_4/conv_8/Variable'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&model/decode_block_4/conv_8/Variable_1'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8
 
 
V
≤	variables
≥trainable_variables
іregularization_losses
µ	keras_api

K0
L1

K0
L1
 
Ю
b	variables
 ґlayer_regularization_losses
Јlayers
ctrainable_variables
Єmetrics
dregularization_losses
єnon_trainable_variables
 

0
 
 
V
Ї	variables
їtrainable_variables
Љregularization_losses
љ	keras_api

M0
N1

M0
N1
 
Ю
k	variables
 Њlayer_regularization_losses
њlayers
ltrainable_variables
јmetrics
mregularization_losses
Ѕnon_trainable_variables
 

0
 
 
V
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api

O0
P1

O0
P1
 
Ю
t	variables
 ∆layer_regularization_losses
«layers
utrainable_variables
»metrics
vregularization_losses
…non_trainable_variables
 

0
 
 
V
 	variables
Ћtrainable_variables
ћregularization_losses
Ќ	keras_api

Q0
R1

Q0
R1
 
Ю
}	variables
 ќlayer_regularization_losses
ѕlayers
~trainable_variables
–metrics
regularization_losses
—non_trainable_variables
 

#0
 
 
V
“	variables
”trainable_variables
‘regularization_losses
’	keras_api

S0
T1

S0
T1
 
°
Ж	variables
 ÷layer_regularization_losses
„layers
Зtrainable_variables
Ўmetrics
Иregularization_losses
ўnon_trainable_variables
 

)0
 
 
V
Џ	variables
џtrainable_variables
№regularization_losses
Ё	keras_api

U0
V1

U0
V1
 
°
П	variables
 ёlayer_regularization_losses
яlayers
Рtrainable_variables
аmetrics
Сregularization_losses
бnon_trainable_variables
 

/0
 
 
V
в	variables
гtrainable_variables
дregularization_losses
е	keras_api

W0
X1

W0
X1
 
°
Ш	variables
 жlayer_regularization_losses
зlayers
Щtrainable_variables
иmetrics
Ъregularization_losses
йnon_trainable_variables
 

50
 
 
V
к	variables
лtrainable_variables
мregularization_losses
н	keras_api

Y0
Z1

Y0
Z1
 
°
°	variables
 оlayer_regularization_losses
пlayers
Ґtrainable_variables
рmetrics
£regularization_losses
сnon_trainable_variables
 

;0
 
 

т	keras_api

[0
\1

[0
\1
 
°
™	variables
 уlayer_regularization_losses
фlayers
Ђtrainable_variables
хmetrics
ђregularization_losses
цnon_trainable_variables
 

A0
 
 
 
 
 
°
≤	variables
 чlayer_regularization_losses
шlayers
≥trainable_variables
щmetrics
іregularization_losses
ъnon_trainable_variables
 

a0
 
 
 
 
 
°
Ї	variables
 ыlayer_regularization_losses
ьlayers
їtrainable_variables
эmetrics
Љregularization_losses
юnon_trainable_variables
 

j0
 
 
 
 
 
°
¬	variables
 €layer_regularization_losses
Аlayers
√trainable_variables
Бmetrics
ƒregularization_losses
Вnon_trainable_variables
 

s0
 
 
 
 
 
°
 	variables
 Гlayer_regularization_losses
Дlayers
Ћtrainable_variables
Еmetrics
ћregularization_losses
Жnon_trainable_variables
 

|0
 
 
 
 
 
°
“	variables
 Зlayer_regularization_losses
Иlayers
”trainable_variables
Йmetrics
‘regularization_losses
Кnon_trainable_variables
 

Е0
 
 
 
 
 
°
Џ	variables
 Лlayer_regularization_losses
Мlayers
џtrainable_variables
Нmetrics
№regularization_losses
Оnon_trainable_variables
 

О0
 
 
 
 
 
°
в	variables
 Пlayer_regularization_losses
Рlayers
гtrainable_variables
Сmetrics
дregularization_losses
Тnon_trainable_variables
 

Ч0
 
 
 
 
 
°
к	variables
 Уlayer_regularization_losses
Фlayers
лtrainable_variables
Хmetrics
мregularization_losses
Цnon_trainable_variables
 

†0
 
 
 
 

©0
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
 
 
 
 
 
 
}
VARIABLE_VALUE'Adam/model/encode_block/conv/Variable/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE)Adam/model/encode_block/conv/Variable/m_1Bvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/model/encode_block_1/conv_1/Variable/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE-Adam/model/encode_block_1/conv_1/Variable/m_1Bvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/model/encode_block_2/conv_2/Variable/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE-Adam/model/encode_block_2/conv_2/Variable/m_1Bvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/model/encode_block_3/conv_3/Variable/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE-Adam/model/encode_block_3/conv_3/Variable/m_1Bvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE)Adam/model/decode_block/conv_4/Variable/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/model/decode_block/conv_4/Variable/m_1Bvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE+Adam/model/decode_block_1/conv_5/Variable/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE-Adam/model/decode_block_1/conv_5/Variable/m_1Cvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE+Adam/model/decode_block_2/conv_6/Variable/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE-Adam/model/decode_block_2/conv_6/Variable/m_1Cvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE+Adam/model/decode_block_3/conv_7/Variable/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE-Adam/model/decode_block_3/conv_7/Variable/m_1Cvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE+Adam/model/decode_block_4/conv_8/Variable/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE-Adam/model/decode_block_4/conv_8/Variable/m_1Cvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/model/encode_block/conv/Variable/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE)Adam/model/encode_block/conv/Variable/v_1Bvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/model/encode_block_1/conv_1/Variable/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE-Adam/model/encode_block_1/conv_1/Variable/v_1Bvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/model/encode_block_2/conv_2/Variable/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE-Adam/model/encode_block_2/conv_2/Variable/v_1Bvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/model/encode_block_3/conv_3/Variable/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE-Adam/model/encode_block_3/conv_3/Variable/v_1Bvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE)Adam/model/decode_block/conv_4/Variable/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/model/decode_block/conv_4/Variable/v_1Bvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE+Adam/model/decode_block_1/conv_5/Variable/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE-Adam/model/decode_block_1/conv_5/Variable/v_1Cvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE+Adam/model/decode_block_2/conv_6/Variable/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE-Adam/model/decode_block_2/conv_6/Variable/v_1Cvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE+Adam/model/decode_block_3/conv_7/Variable/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE-Adam/model/decode_block_3/conv_7/Variable/v_1Cvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE+Adam/model/decode_block_4/conv_8/Variable/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE-Adam/model/decode_block_4/conv_8/Variable/v_1Cvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
О
serving_default_input_1Placeholder*&
shape:€€€€€€€€€иА*
dtype0*1
_output_shapes
:€€€€€€€€€иА
м
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1 model/encode_block/conv/Variable"model/encode_block/conv/Variable_1$model/encode_block_1/conv_1/Variable&model/encode_block_1/conv_1/Variable_1$model/encode_block_2/conv_2/Variable&model/encode_block_2/conv_2/Variable_1$model/encode_block_3/conv_3/Variable&model/encode_block_3/conv_3/Variable_1"model/decode_block/conv_4/Variable$model/decode_block/conv_4/Variable_1$model/decode_block_1/conv_5/Variable&model/decode_block_1/conv_5/Variable_1$model/decode_block_2/conv_6/Variable&model/decode_block_2/conv_6/Variable_1$model/decode_block_3/conv_7/Variable&model/decode_block_3/conv_7/Variable_1$model/decode_block_4/conv_8/Variable&model/decode_block_4/conv_8/Variable_1*,
_gradient_op_typePartitionedCall-14090*,
f'R%
#__inference_signature_wrapper_12988*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иА
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
Ј
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp4model/encode_block/conv/Variable/Read/ReadVariableOp6model/encode_block/conv/Variable_1/Read/ReadVariableOp8model/encode_block_1/conv_1/Variable/Read/ReadVariableOp:model/encode_block_1/conv_1/Variable_1/Read/ReadVariableOp8model/encode_block_2/conv_2/Variable/Read/ReadVariableOp:model/encode_block_2/conv_2/Variable_1/Read/ReadVariableOp8model/encode_block_3/conv_3/Variable/Read/ReadVariableOp:model/encode_block_3/conv_3/Variable_1/Read/ReadVariableOp6model/decode_block/conv_4/Variable/Read/ReadVariableOp8model/decode_block/conv_4/Variable_1/Read/ReadVariableOp8model/decode_block_1/conv_5/Variable/Read/ReadVariableOp:model/decode_block_1/conv_5/Variable_1/Read/ReadVariableOp8model/decode_block_2/conv_6/Variable/Read/ReadVariableOp:model/decode_block_2/conv_6/Variable_1/Read/ReadVariableOp8model/decode_block_3/conv_7/Variable/Read/ReadVariableOp:model/decode_block_3/conv_7/Variable_1/Read/ReadVariableOp8model/decode_block_4/conv_8/Variable/Read/ReadVariableOp:model/decode_block_4/conv_8/Variable_1/Read/ReadVariableOp;Adam/model/encode_block/conv/Variable/m/Read/ReadVariableOp=Adam/model/encode_block/conv/Variable/m_1/Read/ReadVariableOp?Adam/model/encode_block_1/conv_1/Variable/m/Read/ReadVariableOpAAdam/model/encode_block_1/conv_1/Variable/m_1/Read/ReadVariableOp?Adam/model/encode_block_2/conv_2/Variable/m/Read/ReadVariableOpAAdam/model/encode_block_2/conv_2/Variable/m_1/Read/ReadVariableOp?Adam/model/encode_block_3/conv_3/Variable/m/Read/ReadVariableOpAAdam/model/encode_block_3/conv_3/Variable/m_1/Read/ReadVariableOp=Adam/model/decode_block/conv_4/Variable/m/Read/ReadVariableOp?Adam/model/decode_block/conv_4/Variable/m_1/Read/ReadVariableOp?Adam/model/decode_block_1/conv_5/Variable/m/Read/ReadVariableOpAAdam/model/decode_block_1/conv_5/Variable/m_1/Read/ReadVariableOp?Adam/model/decode_block_2/conv_6/Variable/m/Read/ReadVariableOpAAdam/model/decode_block_2/conv_6/Variable/m_1/Read/ReadVariableOp?Adam/model/decode_block_3/conv_7/Variable/m/Read/ReadVariableOpAAdam/model/decode_block_3/conv_7/Variable/m_1/Read/ReadVariableOp?Adam/model/decode_block_4/conv_8/Variable/m/Read/ReadVariableOpAAdam/model/decode_block_4/conv_8/Variable/m_1/Read/ReadVariableOp;Adam/model/encode_block/conv/Variable/v/Read/ReadVariableOp=Adam/model/encode_block/conv/Variable/v_1/Read/ReadVariableOp?Adam/model/encode_block_1/conv_1/Variable/v/Read/ReadVariableOpAAdam/model/encode_block_1/conv_1/Variable/v_1/Read/ReadVariableOp?Adam/model/encode_block_2/conv_2/Variable/v/Read/ReadVariableOpAAdam/model/encode_block_2/conv_2/Variable/v_1/Read/ReadVariableOp?Adam/model/encode_block_3/conv_3/Variable/v/Read/ReadVariableOpAAdam/model/encode_block_3/conv_3/Variable/v_1/Read/ReadVariableOp=Adam/model/decode_block/conv_4/Variable/v/Read/ReadVariableOp?Adam/model/decode_block/conv_4/Variable/v_1/Read/ReadVariableOp?Adam/model/decode_block_1/conv_5/Variable/v/Read/ReadVariableOpAAdam/model/decode_block_1/conv_5/Variable/v_1/Read/ReadVariableOp?Adam/model/decode_block_2/conv_6/Variable/v/Read/ReadVariableOpAAdam/model/decode_block_2/conv_6/Variable/v_1/Read/ReadVariableOp?Adam/model/decode_block_3/conv_7/Variable/v/Read/ReadVariableOpAAdam/model/decode_block_3/conv_7/Variable/v_1/Read/ReadVariableOp?Adam/model/decode_block_4/conv_8/Variable/v/Read/ReadVariableOpAAdam/model/decode_block_4/conv_8/Variable/v_1/Read/ReadVariableOpConst*,
_gradient_op_typePartitionedCall-14171*'
f"R 
__inference__traced_save_14170*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*H
TinA
?2=	*
_output_shapes
: 
Ц
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate model/encode_block/conv/Variable"model/encode_block/conv/Variable_1$model/encode_block_1/conv_1/Variable&model/encode_block_1/conv_1/Variable_1$model/encode_block_2/conv_2/Variable&model/encode_block_2/conv_2/Variable_1$model/encode_block_3/conv_3/Variable&model/encode_block_3/conv_3/Variable_1"model/decode_block/conv_4/Variable$model/decode_block/conv_4/Variable_1$model/decode_block_1/conv_5/Variable&model/decode_block_1/conv_5/Variable_1$model/decode_block_2/conv_6/Variable&model/decode_block_2/conv_6/Variable_1$model/decode_block_3/conv_7/Variable&model/decode_block_3/conv_7/Variable_1$model/decode_block_4/conv_8/Variable&model/decode_block_4/conv_8/Variable_1'Adam/model/encode_block/conv/Variable/m)Adam/model/encode_block/conv/Variable/m_1+Adam/model/encode_block_1/conv_1/Variable/m-Adam/model/encode_block_1/conv_1/Variable/m_1+Adam/model/encode_block_2/conv_2/Variable/m-Adam/model/encode_block_2/conv_2/Variable/m_1+Adam/model/encode_block_3/conv_3/Variable/m-Adam/model/encode_block_3/conv_3/Variable/m_1)Adam/model/decode_block/conv_4/Variable/m+Adam/model/decode_block/conv_4/Variable/m_1+Adam/model/decode_block_1/conv_5/Variable/m-Adam/model/decode_block_1/conv_5/Variable/m_1+Adam/model/decode_block_2/conv_6/Variable/m-Adam/model/decode_block_2/conv_6/Variable/m_1+Adam/model/decode_block_3/conv_7/Variable/m-Adam/model/decode_block_3/conv_7/Variable/m_1+Adam/model/decode_block_4/conv_8/Variable/m-Adam/model/decode_block_4/conv_8/Variable/m_1'Adam/model/encode_block/conv/Variable/v)Adam/model/encode_block/conv/Variable/v_1+Adam/model/encode_block_1/conv_1/Variable/v-Adam/model/encode_block_1/conv_1/Variable/v_1+Adam/model/encode_block_2/conv_2/Variable/v-Adam/model/encode_block_2/conv_2/Variable/v_1+Adam/model/encode_block_3/conv_3/Variable/v-Adam/model/encode_block_3/conv_3/Variable/v_1)Adam/model/decode_block/conv_4/Variable/v+Adam/model/decode_block/conv_4/Variable/v_1+Adam/model/decode_block_1/conv_5/Variable/v-Adam/model/decode_block_1/conv_5/Variable/v_1+Adam/model/decode_block_2/conv_6/Variable/v-Adam/model/decode_block_2/conv_6/Variable/v_1+Adam/model/decode_block_3/conv_7/Variable/v-Adam/model/decode_block_3/conv_7/Variable/v_1+Adam/model/decode_block_4/conv_8/Variable/v-Adam/model/decode_block_4/conv_8/Variable/v_1*,
_gradient_op_typePartitionedCall-14361**
f%R#
!__inference__traced_restore_14360*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*G
Tin@
>2<*
_output_shapes
: ця
ч
l
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_12081

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
БЯ
Ъ
 __inference__wrapped_model_11813
input_1:
6model_encode_block_conv_conv2d_readvariableop_resource7
3model_encode_block_conv_add_readvariableop_resource>
:model_encode_block_1_conv_1_conv2d_readvariableop_resource;
7model_encode_block_1_conv_1_add_readvariableop_resource>
:model_encode_block_2_conv_2_conv2d_readvariableop_resource;
7model_encode_block_2_conv_2_add_readvariableop_resource>
:model_encode_block_3_conv_3_conv2d_readvariableop_resource;
7model_encode_block_3_conv_3_add_readvariableop_resource<
8model_decode_block_conv_4_conv2d_readvariableop_resource9
5model_decode_block_conv_4_add_readvariableop_resource>
:model_decode_block_1_conv_5_conv2d_readvariableop_resource;
7model_decode_block_1_conv_5_add_readvariableop_resource>
:model_decode_block_2_conv_6_conv2d_readvariableop_resource;
7model_decode_block_2_conv_6_add_readvariableop_resource>
:model_decode_block_3_conv_7_conv2d_readvariableop_resource;
7model_decode_block_3_conv_7_add_readvariableop_resource>
:model_decode_block_4_conv_8_conv2d_readvariableop_resource;
7model_decode_block_4_conv_8_add_readvariableop_resource
identityИҐ,model/decode_block/conv_4/Add/ReadVariableOpҐ/model/decode_block/conv_4/Conv2D/ReadVariableOpҐ.model/decode_block_1/conv_5/Add/ReadVariableOpҐ1model/decode_block_1/conv_5/Conv2D/ReadVariableOpҐ.model/decode_block_2/conv_6/Add/ReadVariableOpҐ1model/decode_block_2/conv_6/Conv2D/ReadVariableOpҐ.model/decode_block_3/conv_7/Add/ReadVariableOpҐ1model/decode_block_3/conv_7/Conv2D/ReadVariableOpҐ.model/decode_block_4/conv_8/Add/ReadVariableOpҐ1model/decode_block_4/conv_8/Conv2D/ReadVariableOpҐ*model/encode_block/conv/Add/ReadVariableOpҐ-model/encode_block/conv/Conv2D/ReadVariableOpҐ.model/encode_block_1/conv_1/Add/ReadVariableOpҐ1model/encode_block_1/conv_1/Conv2D/ReadVariableOpҐ.model/encode_block_2/conv_2/Add/ReadVariableOpҐ1model/encode_block_2/conv_2/Conv2D/ReadVariableOpҐ.model/encode_block_3/conv_3/Add/ReadVariableOpҐ1model/encode_block_3/conv_3/Conv2D/ReadVariableOpЏ
-model/encode_block/conv/Conv2D/ReadVariableOpReadVariableOp6model_encode_block_conv_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ћ
model/encode_block/conv/Conv2DConv2Dinput_15model/encode_block/conv/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іј@»
*model/encode_block/conv/Add/ReadVariableOpReadVariableOp3model_encode_block_conv_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ї
model/encode_block/conv/AddAdd'model/encode_block/conv/Conv2D:output:02model/encode_block/conv/Add/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€іј@Б
model/encode_block/conv/ReluRelumodel/encode_block/conv/Add:z:0*
T0*1
_output_shapes
:€€€€€€€€€іј@¶
2model/encode_block/conv/spatial_dropout2d/IdentityIdentity*model/encode_block/conv/Relu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€іј@Ь
model/encode_block/MaxPoolMaxPoolinput_1*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іјi
model/encode_block/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: х
model/encode_block/concatConcatV2;model/encode_block/conv/spatial_dropout2d/Identity:output:0#model/encode_block/MaxPool:output:0'model/encode_block/concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€іјCв
1model/encode_block_1/conv_1/Conv2D/ReadVariableOpReadVariableOp:model_encode_block_1_conv_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@З
"model/encode_block_1/conv_1/Conv2DConv2D;model/encode_block/conv/spatial_dropout2d/Identity:output:09model/encode_block_1/conv_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@–
.model/encode_block_1/conv_1/Add/ReadVariableOpReadVariableOp7model_encode_block_1_conv_1_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@∆
model/encode_block_1/conv_1/AddAdd+model/encode_block_1/conv_1/Conv2D:output:06model/encode_block_1/conv_1/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@И
 model/encode_block_1/conv_1/ReluRelu#model/encode_block_1/conv_1/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@ѓ
8model/encode_block_1/conv_1/spatial_dropout2d_1/IdentityIdentity.model/encode_block_1/conv_1/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@—
model/encode_block_1/MaxPoolMaxPool;model/encode_block/conv/spatial_dropout2d/Identity:output:0*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@k
 model/encode_block_1/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Б
model/encode_block_1/concatConcatV2Amodel/encode_block_1/conv_1/spatial_dropout2d_1/Identity:output:0%model/encode_block_1/MaxPool:output:0)model/encode_block_1/concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€Z†Ав
1model/encode_block_2/conv_2/Conv2D/ReadVariableOpReadVariableOp:model_encode_block_2_conv_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@М
"model/encode_block_2/conv_2/Conv2DConv2DAmodel/encode_block_1/conv_1/spatial_dropout2d_1/Identity:output:09model/encode_block_2/conv_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@–
.model/encode_block_2/conv_2/Add/ReadVariableOpReadVariableOp7model_encode_block_2_conv_2_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@≈
model/encode_block_2/conv_2/AddAdd+model/encode_block_2/conv_2/Conv2D:output:06model/encode_block_2/conv_2/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P@З
 model/encode_block_2/conv_2/ReluRelu#model/encode_block_2/conv_2/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P@Ѓ
8model/encode_block_2/conv_2/spatial_dropout2d_2/IdentityIdentity.model/encode_block_2/conv_2/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P@÷
model/encode_block_2/MaxPoolMaxPoolAmodel/encode_block_1/conv_1/spatial_dropout2d_1/Identity:output:0*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@k
 model/encode_block_2/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: А
model/encode_block_2/concatConcatV2Amodel/encode_block_2/conv_2/spatial_dropout2d_2/Identity:output:0%model/encode_block_2/MaxPool:output:0)model/encode_block_2/concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€-PАв
1model/encode_block_3/conv_3/Conv2D/ReadVariableOpReadVariableOp:model_encode_block_3_conv_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@М
"model/encode_block_3/conv_3/Conv2DConv2DAmodel/encode_block_2/conv_2/spatial_dropout2d_2/Identity:output:09model/encode_block_3/conv_3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@–
.model/encode_block_3/conv_3/Add/ReadVariableOpReadVariableOp7model_encode_block_3_conv_3_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@≈
model/encode_block_3/conv_3/AddAdd+model/encode_block_3/conv_3/Conv2D:output:06model/encode_block_3/conv_3/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€(@З
 model/encode_block_3/conv_3/ReluRelu#model/encode_block_3/conv_3/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€(@Ѓ
8model/encode_block_3/conv_3/spatial_dropout2d_3/IdentityIdentity.model/encode_block_3/conv_3/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€(@÷
model/encode_block_3/MaxPoolMaxPoolAmodel/encode_block_2/conv_2/spatial_dropout2d_2/Identity:output:0*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@k
 model/encode_block_3/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: А
model/encode_block_3/concatConcatV2Amodel/encode_block_3/conv_3/spatial_dropout2d_3/Identity:output:0%model/encode_block_3/MaxPool:output:0)model/encode_block_3/concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€(Аё
/model/decode_block/conv_4/Conv2D/ReadVariableOpReadVariableOp8model_decode_block_conv_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ И
 model/decode_block/conv_4/Conv2DConv2DAmodel/encode_block_3/conv_3/spatial_dropout2d_3/Identity:output:07model/decode_block/conv_4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€( ћ
,model/decode_block/conv_4/Add/ReadVariableOpReadVariableOp5model_decode_block_conv_4_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: њ
model/decode_block/conv_4/AddAdd)model/decode_block/conv_4/Conv2D:output:04model/decode_block/conv_4/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€( Г
model/decode_block/conv_4/ReluRelu!model/decode_block/conv_4/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€( ™
6model/decode_block/conv_4/spatial_dropout2d_4/IdentityIdentity,model/decode_block/conv_4/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€( o
model/decode_block/resize/sizeConst*
valueB"-   P   *
dtype0*
_output_shapes
:ш
(model/decode_block/resize/ResizeBilinearResizeBilinear?model/decode_block/conv_4/spatial_dropout2d_4/Identity:output:0'model/decode_block/resize/size:output:0*
half_pixel_centers(*
T0*/
_output_shapes
:€€€€€€€€€-P в
1model/decode_block_1/conv_5/Conv2D/ReadVariableOpReadVariableOp:model_decode_block_1_conv_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  Д
"model/decode_block_1/conv_5/Conv2DConv2D9model/decode_block/resize/ResizeBilinear:resized_images:09model/decode_block_1/conv_5/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P –
.model/decode_block_1/conv_5/Add/ReadVariableOpReadVariableOp7model_decode_block_1_conv_5_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ≈
model/decode_block_1/conv_5/AddAdd+model/decode_block_1/conv_5/Conv2D:output:06model/decode_block_1/conv_5/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P З
 model/decode_block_1/conv_5/ReluRelu#model/decode_block_1/conv_5/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P Ѓ
8model/decode_block_1/conv_5/spatial_dropout2d_5/IdentityIdentity.model/decode_block_1/conv_5/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P q
 model/decode_block_1/resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:€
*model/decode_block_1/resize/ResizeBilinearResizeBilinearAmodel/decode_block_1/conv_5/spatial_dropout2d_5/Identity:output:0)model/decode_block_1/resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† в
1model/decode_block_2/conv_6/Conv2D/ReadVariableOpReadVariableOp:model_decode_block_2_conv_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ М
"model/decode_block_2/conv_6/Conv2DConv2DAmodel/encode_block_2/conv_2/spatial_dropout2d_2/Identity:output:09model/decode_block_2/conv_6/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P –
.model/decode_block_2/conv_6/Add/ReadVariableOpReadVariableOp7model_decode_block_2_conv_6_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ≈
model/decode_block_2/conv_6/AddAdd+model/decode_block_2/conv_6/Conv2D:output:06model/decode_block_2/conv_6/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P З
 model/decode_block_2/conv_6/ReluRelu#model/decode_block_2/conv_6/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P Ѓ
8model/decode_block_2/conv_6/spatial_dropout2d_6/IdentityIdentity.model/decode_block_2/conv_6/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P q
 model/decode_block_2/resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:€
*model/decode_block_2/resize/ResizeBilinearResizeBilinearAmodel/decode_block_2/conv_6/spatial_dropout2d_6/Identity:output:0)model/decode_block_2/resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† в
1model/decode_block_3/conv_7/Conv2D/ReadVariableOpReadVariableOp:model_decode_block_3_conv_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ Н
"model/decode_block_3/conv_7/Conv2DConv2DAmodel/encode_block_1/conv_1/spatial_dropout2d_1/Identity:output:09model/decode_block_3/conv_7/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z† –
.model/decode_block_3/conv_7/Add/ReadVariableOpReadVariableOp7model_decode_block_3_conv_7_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ∆
model/decode_block_3/conv_7/AddAdd+model/decode_block_3/conv_7/Conv2D:output:06model/decode_block_3/conv_7/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z† И
 model/decode_block_3/conv_7/ReluRelu#model/decode_block_3/conv_7/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z† ѓ
8model/decode_block_3/conv_7/spatial_dropout2d_7/IdentityIdentity.model/decode_block_3/conv_7/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€Z† q
 model/decode_block_3/resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:€
*model/decode_block_3/resize/ResizeBilinearResizeBilinearAmodel/decode_block_3/conv_7/spatial_dropout2d_7/Identity:output:0)model/decode_block_3/resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† \
model/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ѓ
model/concatConcatV2;model/decode_block_1/resize/ResizeBilinear:resized_images:0;model/decode_block_2/resize/ResizeBilinear:resized_images:0;model/decode_block_3/resize/ResizeBilinear:resized_images:0model/concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€Z†`в
1model/decode_block_4/conv_8/Conv2D/ReadVariableOpReadVariableOp:model_decode_block_4_conv_8_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:`б
"model/decode_block_4/conv_8/Conv2DConv2Dmodel/concat:output:09model/decode_block_4/conv_8/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†–
.model/decode_block_4/conv_8/Add/ReadVariableOpReadVariableOp7model_decode_block_4_conv_8_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:∆
model/decode_block_4/conv_8/AddAdd+model/decode_block_4/conv_8/Conv2D:output:06model/decode_block_4/conv_8/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†q
 model/decode_block_4/resize/sizeConst*
valueB"h  А  *
dtype0*
_output_shapes
:в
*model/decode_block_4/resize/ResizeBilinearResizeBilinear#model/decode_block_4/conv_8/Add:z:0)model/decode_block_4/resize/size:output:0*
half_pixel_centers(*
T0*1
_output_shapes
:€€€€€€€€€иАС
model/SigmoidSigmoid;model/decode_block_4/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:€€€€€€€€€иАд
IdentityIdentitymodel/Sigmoid:y:0-^model/decode_block/conv_4/Add/ReadVariableOp0^model/decode_block/conv_4/Conv2D/ReadVariableOp/^model/decode_block_1/conv_5/Add/ReadVariableOp2^model/decode_block_1/conv_5/Conv2D/ReadVariableOp/^model/decode_block_2/conv_6/Add/ReadVariableOp2^model/decode_block_2/conv_6/Conv2D/ReadVariableOp/^model/decode_block_3/conv_7/Add/ReadVariableOp2^model/decode_block_3/conv_7/Conv2D/ReadVariableOp/^model/decode_block_4/conv_8/Add/ReadVariableOp2^model/decode_block_4/conv_8/Conv2D/ReadVariableOp+^model/encode_block/conv/Add/ReadVariableOp.^model/encode_block/conv/Conv2D/ReadVariableOp/^model/encode_block_1/conv_1/Add/ReadVariableOp2^model/encode_block_1/conv_1/Conv2D/ReadVariableOp/^model/encode_block_2/conv_2/Add/ReadVariableOp2^model/encode_block_2/conv_2/Conv2D/ReadVariableOp/^model/encode_block_3/conv_3/Add/ReadVariableOp2^model/encode_block_3/conv_3/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::2f
1model/decode_block_2/conv_6/Conv2D/ReadVariableOp1model/decode_block_2/conv_6/Conv2D/ReadVariableOp2f
1model/encode_block_3/conv_3/Conv2D/ReadVariableOp1model/encode_block_3/conv_3/Conv2D/ReadVariableOp2`
.model/decode_block_4/conv_8/Add/ReadVariableOp.model/decode_block_4/conv_8/Add/ReadVariableOp2f
1model/encode_block_2/conv_2/Conv2D/ReadVariableOp1model/encode_block_2/conv_2/Conv2D/ReadVariableOp2f
1model/decode_block_1/conv_5/Conv2D/ReadVariableOp1model/decode_block_1/conv_5/Conv2D/ReadVariableOp2f
1model/encode_block_1/conv_1/Conv2D/ReadVariableOp1model/encode_block_1/conv_1/Conv2D/ReadVariableOp2^
-model/encode_block/conv/Conv2D/ReadVariableOp-model/encode_block/conv/Conv2D/ReadVariableOp2\
,model/decode_block/conv_4/Add/ReadVariableOp,model/decode_block/conv_4/Add/ReadVariableOp2`
.model/encode_block_1/conv_1/Add/ReadVariableOp.model/encode_block_1/conv_1/Add/ReadVariableOp2X
*model/encode_block/conv/Add/ReadVariableOp*model/encode_block/conv/Add/ReadVariableOp2`
.model/decode_block_3/conv_7/Add/ReadVariableOp.model/decode_block_3/conv_7/Add/ReadVariableOp2b
/model/decode_block/conv_4/Conv2D/ReadVariableOp/model/decode_block/conv_4/Conv2D/ReadVariableOp2`
.model/encode_block_3/conv_3/Add/ReadVariableOp.model/encode_block_3/conv_3/Add/ReadVariableOp2`
.model/decode_block_2/conv_6/Add/ReadVariableOp.model/decode_block_2/conv_6/Add/ReadVariableOp2f
1model/decode_block_4/conv_8/Conv2D/ReadVariableOp1model/decode_block_4/conv_8/Conv2D/ReadVariableOp2f
1model/decode_block_3/conv_7/Conv2D/ReadVariableOp1model/decode_block_3/conv_7/Conv2D/ReadVariableOp2`
.model/encode_block_2/conv_2/Add/ReadVariableOp.model/encode_block_2/conv_2/Add/ReadVariableOp2`
.model/decode_block_1/conv_5/Add/ReadVariableOp.model/decode_block_1/conv_5/Add/ReadVariableOp: : : :' #
!
_user_specified_name	input_1: : : : :
 : : : : : : : : :	 : 
‘
ъ
G__inference_decode_block_layer_call_and_return_conditional_losses_13528
input_tensor)
%conv_4_conv2d_readvariableop_resource&
"conv_4_add_readvariableop_resource
identityИҐconv_4/Add/ReadVariableOpҐconv_4/Conv2D/ReadVariableOpЄ
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ≠
conv_4/Conv2DConv2Dinput_tensor$conv_4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€( ¶
conv_4/Add/ReadVariableOpReadVariableOp"conv_4_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_4/AddAddconv_4/Conv2D:output:0!conv_4/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€( ]
conv_4/ReluReluconv_4/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€( i
 conv_4/spatial_dropout2d_4/ShapeShapeconv_4/Relu:activations:0*
T0*
_output_shapes
:x
.conv_4/spatial_dropout2d_4/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_4/spatial_dropout2d_4/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_4/spatial_dropout2d_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_4/spatial_dropout2d_4/strided_sliceStridedSlice)conv_4/spatial_dropout2d_4/Shape:output:07conv_4/spatial_dropout2d_4/strided_slice/stack:output:09conv_4/spatial_dropout2d_4/strided_slice/stack_1:output:09conv_4/spatial_dropout2d_4/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_4/spatial_dropout2d_4/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_4/spatial_dropout2d_4/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_4/spatial_dropout2d_4/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_4/spatial_dropout2d_4/strided_slice_1StridedSlice)conv_4/spatial_dropout2d_4/Shape:output:09conv_4/spatial_dropout2d_4/strided_slice_1/stack:output:0;conv_4/spatial_dropout2d_4/strided_slice_1/stack_1:output:0;conv_4/spatial_dropout2d_4/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: \
resize/sizeConst*
valueB"-   P   *
dtype0*
_output_shapes
:ђ
resize/ResizeBilinearResizeBilinearconv_4/Relu:activations:0resize/size:output:0*
half_pixel_centers(*
T0*/
_output_shapes
:€€€€€€€€€-P ±
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_4/Add/ReadVariableOp^conv_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€-P "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€(@::2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp26
conv_4/Add/ReadVariableOpconv_4/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
÷
т
G__inference_encode_block_layer_call_and_return_conditional_losses_13321
input_tensor'
#conv_conv2d_readvariableop_resource$
 conv_add_readvariableop_resource
identityИҐconv/Add/ReadVariableOpҐconv/Conv2D/ReadVariableOpі
conv/Conv2D/ReadVariableOpReadVariableOp#conv_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@Ђ
conv/Conv2DConv2Dinput_tensor"conv/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іј@Ґ
conv/Add/ReadVariableOpReadVariableOp conv_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@В
conv/AddAddconv/Conv2D:output:0conv/Add/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€іј@[
	conv/ReluReluconv/Add:z:0*
T0*1
_output_shapes
:€€€€€€€€€іј@c
conv/spatial_dropout2d/ShapeShapeconv/Relu:activations:0*
T0*
_output_shapes
:t
*conv/spatial_dropout2d/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:v
,conv/spatial_dropout2d/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:v
,conv/spatial_dropout2d/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ƒ
$conv/spatial_dropout2d/strided_sliceStridedSlice%conv/spatial_dropout2d/Shape:output:03conv/spatial_dropout2d/strided_slice/stack:output:05conv/spatial_dropout2d/strided_slice/stack_1:output:05conv/spatial_dropout2d/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: v
,conv/spatial_dropout2d/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:x
.conv/spatial_dropout2d/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:x
.conv/spatial_dropout2d/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ћ
&conv/spatial_dropout2d/strided_slice_1StridedSlice%conv/spatial_dropout2d/Shape:output:05conv/spatial_dropout2d/strided_slice_1/stack:output:07conv/spatial_dropout2d/strided_slice_1/stack_1:output:07conv/spatial_dropout2d/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: О
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іјV
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Ш
concatConcatV2conv/Relu:activations:0MaxPool:output:0concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€іјC†
IdentityIdentityconv/Relu:activations:0^conv/Add/ReadVariableOp^conv/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€іј@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€иА::28
conv/Conv2D/ReadVariableOpconv/Conv2D/ReadVariableOp22
conv/Add/ReadVariableOpconv/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
ол
ї(
!__inference__traced_restore_14360
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate7
3assignvariableop_5_model_encode_block_conv_variable9
5assignvariableop_6_model_encode_block_conv_variable_1;
7assignvariableop_7_model_encode_block_1_conv_1_variable=
9assignvariableop_8_model_encode_block_1_conv_1_variable_1;
7assignvariableop_9_model_encode_block_2_conv_2_variable>
:assignvariableop_10_model_encode_block_2_conv_2_variable_1<
8assignvariableop_11_model_encode_block_3_conv_3_variable>
:assignvariableop_12_model_encode_block_3_conv_3_variable_1:
6assignvariableop_13_model_decode_block_conv_4_variable<
8assignvariableop_14_model_decode_block_conv_4_variable_1<
8assignvariableop_15_model_decode_block_1_conv_5_variable>
:assignvariableop_16_model_decode_block_1_conv_5_variable_1<
8assignvariableop_17_model_decode_block_2_conv_6_variable>
:assignvariableop_18_model_decode_block_2_conv_6_variable_1<
8assignvariableop_19_model_decode_block_3_conv_7_variable>
:assignvariableop_20_model_decode_block_3_conv_7_variable_1<
8assignvariableop_21_model_decode_block_4_conv_8_variable>
:assignvariableop_22_model_decode_block_4_conv_8_variable_1?
;assignvariableop_23_adam_model_encode_block_conv_variable_mA
=assignvariableop_24_adam_model_encode_block_conv_variable_m_1C
?assignvariableop_25_adam_model_encode_block_1_conv_1_variable_mE
Aassignvariableop_26_adam_model_encode_block_1_conv_1_variable_m_1C
?assignvariableop_27_adam_model_encode_block_2_conv_2_variable_mE
Aassignvariableop_28_adam_model_encode_block_2_conv_2_variable_m_1C
?assignvariableop_29_adam_model_encode_block_3_conv_3_variable_mE
Aassignvariableop_30_adam_model_encode_block_3_conv_3_variable_m_1A
=assignvariableop_31_adam_model_decode_block_conv_4_variable_mC
?assignvariableop_32_adam_model_decode_block_conv_4_variable_m_1C
?assignvariableop_33_adam_model_decode_block_1_conv_5_variable_mE
Aassignvariableop_34_adam_model_decode_block_1_conv_5_variable_m_1C
?assignvariableop_35_adam_model_decode_block_2_conv_6_variable_mE
Aassignvariableop_36_adam_model_decode_block_2_conv_6_variable_m_1C
?assignvariableop_37_adam_model_decode_block_3_conv_7_variable_mE
Aassignvariableop_38_adam_model_decode_block_3_conv_7_variable_m_1C
?assignvariableop_39_adam_model_decode_block_4_conv_8_variable_mE
Aassignvariableop_40_adam_model_decode_block_4_conv_8_variable_m_1?
;assignvariableop_41_adam_model_encode_block_conv_variable_vA
=assignvariableop_42_adam_model_encode_block_conv_variable_v_1C
?assignvariableop_43_adam_model_encode_block_1_conv_1_variable_vE
Aassignvariableop_44_adam_model_encode_block_1_conv_1_variable_v_1C
?assignvariableop_45_adam_model_encode_block_2_conv_2_variable_vE
Aassignvariableop_46_adam_model_encode_block_2_conv_2_variable_v_1C
?assignvariableop_47_adam_model_encode_block_3_conv_3_variable_vE
Aassignvariableop_48_adam_model_encode_block_3_conv_3_variable_v_1A
=assignvariableop_49_adam_model_decode_block_conv_4_variable_vC
?assignvariableop_50_adam_model_decode_block_conv_4_variable_v_1C
?assignvariableop_51_adam_model_decode_block_1_conv_5_variable_vE
Aassignvariableop_52_adam_model_decode_block_1_conv_5_variable_v_1C
?assignvariableop_53_adam_model_decode_block_2_conv_6_variable_vE
Aassignvariableop_54_adam_model_decode_block_2_conv_6_variable_v_1C
?assignvariableop_55_adam_model_decode_block_3_conv_7_variable_vE
Aassignvariableop_56_adam_model_decode_block_3_conv_7_variable_v_1C
?assignvariableop_57_adam_model_decode_block_4_conv_8_variable_vE
Aassignvariableop_58_adam_model_decode_block_4_conv_8_variable_v_1
identity_60ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1ћ
RestoreV2/tensor_namesConst"/device:CPU:0*т
valueиBе;B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:;и
RestoreV2/shape_and_slicesConst"/device:CPU:0*К
valueАB~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:;»
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*I
dtypes?
=2;	*В
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:v
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0*
dtype0	*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:~
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:~
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:}
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:Е
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:У
AssignVariableOp_5AssignVariableOp3assignvariableop_5_model_encode_block_conv_variableIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:Х
AssignVariableOp_6AssignVariableOp5assignvariableop_6_model_encode_block_conv_variable_1Identity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:Ч
AssignVariableOp_7AssignVariableOp7assignvariableop_7_model_encode_block_1_conv_1_variableIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:Щ
AssignVariableOp_8AssignVariableOp9assignvariableop_8_model_encode_block_1_conv_1_variable_1Identity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:Ч
AssignVariableOp_9AssignVariableOp7assignvariableop_9_model_encode_block_2_conv_2_variableIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:Ь
AssignVariableOp_10AssignVariableOp:assignvariableop_10_model_encode_block_2_conv_2_variable_1Identity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:Ъ
AssignVariableOp_11AssignVariableOp8assignvariableop_11_model_encode_block_3_conv_3_variableIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:Ь
AssignVariableOp_12AssignVariableOp:assignvariableop_12_model_encode_block_3_conv_3_variable_1Identity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:Ш
AssignVariableOp_13AssignVariableOp6assignvariableop_13_model_decode_block_conv_4_variableIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:Ъ
AssignVariableOp_14AssignVariableOp8assignvariableop_14_model_decode_block_conv_4_variable_1Identity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:Ъ
AssignVariableOp_15AssignVariableOp8assignvariableop_15_model_decode_block_1_conv_5_variableIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:Ь
AssignVariableOp_16AssignVariableOp:assignvariableop_16_model_decode_block_1_conv_5_variable_1Identity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:Ъ
AssignVariableOp_17AssignVariableOp8assignvariableop_17_model_decode_block_2_conv_6_variableIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:Ь
AssignVariableOp_18AssignVariableOp:assignvariableop_18_model_decode_block_2_conv_6_variable_1Identity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:Ъ
AssignVariableOp_19AssignVariableOp8assignvariableop_19_model_decode_block_3_conv_7_variableIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:Ь
AssignVariableOp_20AssignVariableOp:assignvariableop_20_model_decode_block_3_conv_7_variable_1Identity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:Ъ
AssignVariableOp_21AssignVariableOp8assignvariableop_21_model_decode_block_4_conv_8_variableIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:Ь
AssignVariableOp_22AssignVariableOp:assignvariableop_22_model_decode_block_4_conv_8_variable_1Identity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:Э
AssignVariableOp_23AssignVariableOp;assignvariableop_23_adam_model_encode_block_conv_variable_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:Я
AssignVariableOp_24AssignVariableOp=assignvariableop_24_adam_model_encode_block_conv_variable_m_1Identity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:°
AssignVariableOp_25AssignVariableOp?assignvariableop_25_adam_model_encode_block_1_conv_1_variable_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:£
AssignVariableOp_26AssignVariableOpAassignvariableop_26_adam_model_encode_block_1_conv_1_variable_m_1Identity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:°
AssignVariableOp_27AssignVariableOp?assignvariableop_27_adam_model_encode_block_2_conv_2_variable_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:£
AssignVariableOp_28AssignVariableOpAassignvariableop_28_adam_model_encode_block_2_conv_2_variable_m_1Identity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:°
AssignVariableOp_29AssignVariableOp?assignvariableop_29_adam_model_encode_block_3_conv_3_variable_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:£
AssignVariableOp_30AssignVariableOpAassignvariableop_30_adam_model_encode_block_3_conv_3_variable_m_1Identity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:Я
AssignVariableOp_31AssignVariableOp=assignvariableop_31_adam_model_decode_block_conv_4_variable_mIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:°
AssignVariableOp_32AssignVariableOp?assignvariableop_32_adam_model_decode_block_conv_4_variable_m_1Identity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:°
AssignVariableOp_33AssignVariableOp?assignvariableop_33_adam_model_decode_block_1_conv_5_variable_mIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:£
AssignVariableOp_34AssignVariableOpAassignvariableop_34_adam_model_decode_block_1_conv_5_variable_m_1Identity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:°
AssignVariableOp_35AssignVariableOp?assignvariableop_35_adam_model_decode_block_2_conv_6_variable_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:£
AssignVariableOp_36AssignVariableOpAassignvariableop_36_adam_model_decode_block_2_conv_6_variable_m_1Identity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:°
AssignVariableOp_37AssignVariableOp?assignvariableop_37_adam_model_decode_block_3_conv_7_variable_mIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:£
AssignVariableOp_38AssignVariableOpAassignvariableop_38_adam_model_decode_block_3_conv_7_variable_m_1Identity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:°
AssignVariableOp_39AssignVariableOp?assignvariableop_39_adam_model_decode_block_4_conv_8_variable_mIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:£
AssignVariableOp_40AssignVariableOpAassignvariableop_40_adam_model_decode_block_4_conv_8_variable_m_1Identity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:Э
AssignVariableOp_41AssignVariableOp;assignvariableop_41_adam_model_encode_block_conv_variable_vIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:Я
AssignVariableOp_42AssignVariableOp=assignvariableop_42_adam_model_encode_block_conv_variable_v_1Identity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:°
AssignVariableOp_43AssignVariableOp?assignvariableop_43_adam_model_encode_block_1_conv_1_variable_vIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:£
AssignVariableOp_44AssignVariableOpAassignvariableop_44_adam_model_encode_block_1_conv_1_variable_v_1Identity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:°
AssignVariableOp_45AssignVariableOp?assignvariableop_45_adam_model_encode_block_2_conv_2_variable_vIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:£
AssignVariableOp_46AssignVariableOpAassignvariableop_46_adam_model_encode_block_2_conv_2_variable_v_1Identity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:°
AssignVariableOp_47AssignVariableOp?assignvariableop_47_adam_model_encode_block_3_conv_3_variable_vIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:£
AssignVariableOp_48AssignVariableOpAassignvariableop_48_adam_model_encode_block_3_conv_3_variable_v_1Identity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:Я
AssignVariableOp_49AssignVariableOp=assignvariableop_49_adam_model_decode_block_conv_4_variable_vIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:°
AssignVariableOp_50AssignVariableOp?assignvariableop_50_adam_model_decode_block_conv_4_variable_v_1Identity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:°
AssignVariableOp_51AssignVariableOp?assignvariableop_51_adam_model_decode_block_1_conv_5_variable_vIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:£
AssignVariableOp_52AssignVariableOpAassignvariableop_52_adam_model_decode_block_1_conv_5_variable_v_1Identity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:°
AssignVariableOp_53AssignVariableOp?assignvariableop_53_adam_model_decode_block_2_conv_6_variable_vIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:£
AssignVariableOp_54AssignVariableOpAassignvariableop_54_adam_model_decode_block_2_conv_6_variable_v_1Identity_54:output:0*
dtype0*
_output_shapes
 P
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:°
AssignVariableOp_55AssignVariableOp?assignvariableop_55_adam_model_decode_block_3_conv_7_variable_vIdentity_55:output:0*
dtype0*
_output_shapes
 P
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:£
AssignVariableOp_56AssignVariableOpAassignvariableop_56_adam_model_decode_block_3_conv_7_variable_v_1Identity_56:output:0*
dtype0*
_output_shapes
 P
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:°
AssignVariableOp_57AssignVariableOp?assignvariableop_57_adam_model_decode_block_4_conv_8_variable_vIdentity_57:output:0*
dtype0*
_output_shapes
 P
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:£
AssignVariableOp_58AssignVariableOpAassignvariableop_58_adam_model_decode_block_4_conv_8_variable_v_1Identity_58:output:0*
dtype0*
_output_shapes
 М
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:µ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 б

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: о

Identity_60IdentityIdentity_59:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_60Identity_60:output:0*Г
_input_shapesс
о: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_58: : :1 :  : : :9 :( : : :0 :# : :	 :8 :+ : :+ '
%
_user_specified_namefile_prefix:3 :" : : :; :* :% : : :2 :- : : :: :5 :$ : : :, : :
 : :4 :' : : :/ : : : :7 :& : : :. : : :6 :! : : :) 
¶
j
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_13897

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
№
ь
I__inference_decode_block_3_layer_call_and_return_conditional_losses_12708
input_tensor)
%conv_7_conv2d_readvariableop_resource&
"conv_7_add_readvariableop_resource
identityИҐconv_7/Add/ReadVariableOpҐconv_7/Conv2D/ReadVariableOpЄ
conv_7/Conv2D/ReadVariableOpReadVariableOp%conv_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ Ѓ
conv_7/Conv2DConv2Dinput_tensor$conv_7/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z† ¶
conv_7/Add/ReadVariableOpReadVariableOp"conv_7_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: З

conv_7/AddAddconv_7/Conv2D:output:0!conv_7/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z† ^
conv_7/ReluReluconv_7/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z† i
 conv_7/spatial_dropout2d_7/ShapeShapeconv_7/Relu:activations:0*
T0*
_output_shapes
:x
.conv_7/spatial_dropout2d_7/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_7/spatial_dropout2d_7/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_7/spatial_dropout2d_7/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_7/spatial_dropout2d_7/strided_sliceStridedSlice)conv_7/spatial_dropout2d_7/Shape:output:07conv_7/spatial_dropout2d_7/strided_slice/stack:output:09conv_7/spatial_dropout2d_7/strided_slice/stack_1:output:09conv_7/spatial_dropout2d_7/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_7/spatial_dropout2d_7/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_7/spatial_dropout2d_7/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_7/spatial_dropout2d_7/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_7/spatial_dropout2d_7/strided_slice_1StridedSlice)conv_7/spatial_dropout2d_7/Shape:output:09conv_7/spatial_dropout2d_7/strided_slice_1/stack:output:0;conv_7/spatial_dropout2d_7/strided_slice_1/stack_1:output:0;conv_7/spatial_dropout2d_7/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:≠
resize/ResizeBilinearResizeBilinearconv_7/Relu:activations:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_7/Add/ReadVariableOp^conv_7/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::2<
conv_7/Conv2D/ReadVariableOpconv_7/Conv2D/ReadVariableOp26
conv_7/Add/ReadVariableOpconv_7/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Ј
O
3__inference_spatial_dropout2d_7_layer_call_fn_13968

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12244*W
fRRP
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_12243*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ј
O
3__inference_spatial_dropout2d_7_layer_call_fn_13963

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12231*W
fRRP
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_12230*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
њ
ь
I__inference_encode_block_3_layer_call_and_return_conditional_losses_13477
input_tensor)
%conv_3_conv2d_readvariableop_resource&
"conv_3_add_readvariableop_resource
identityИҐconv_3/Add/ReadVariableOpҐconv_3/Conv2D/ReadVariableOpЄ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@≠
conv_3/Conv2DConv2Dinput_tensor$conv_3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@¶
conv_3/Add/ReadVariableOpReadVariableOp"conv_3_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ж

conv_3/AddAddconv_3/Conv2D:output:0!conv_3/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€(@]
conv_3/ReluReluconv_3/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€(@i
 conv_3/spatial_dropout2d_3/ShapeShapeconv_3/Relu:activations:0*
T0*
_output_shapes
:x
.conv_3/spatial_dropout2d_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_3/spatial_dropout2d_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_3/spatial_dropout2d_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_3/spatial_dropout2d_3/strided_sliceStridedSlice)conv_3/spatial_dropout2d_3/Shape:output:07conv_3/spatial_dropout2d_3/strided_slice/stack:output:09conv_3/spatial_dropout2d_3/strided_slice/stack_1:output:09conv_3/spatial_dropout2d_3/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_3/spatial_dropout2d_3/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_3/spatial_dropout2d_3/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_3/spatial_dropout2d_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_3/spatial_dropout2d_3/strided_slice_1StridedSlice)conv_3/spatial_dropout2d_3/Shape:output:09conv_3/spatial_dropout2d_3/strided_slice_1/stack:output:0;conv_3/spatial_dropout2d_3/strided_slice_1/stack_1:output:0;conv_3/spatial_dropout2d_3/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: М
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Щ
concatConcatV2conv_3/Relu:activations:0MaxPool:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€(А§
IdentityIdentityconv_3/Relu:activations:0^conv_3/Add/ReadVariableOp^conv_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€(@"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::26
conv_3/Add/ReadVariableOpconv_3/Add/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
М
≥
,__inference_decode_block_layer_call_fn_13549
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12545*P
fKRI
G__inference_decode_block_layer_call_and_return_conditional_losses_12525*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P К
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€-P "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€(@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
’
о
#__inference_signature_wrapper_12988
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
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*,
_gradient_op_typePartitionedCall-12967*)
f$R"
 __inference__wrapped_model_11813*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иАМ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :' #
!
_user_specified_name	input_1: : : : :
 : : : : : : : : :	 : 
лТ
з
@__inference_model_layer_call_and_return_conditional_losses_13252
input_tensor4
0encode_block_conv_conv2d_readvariableop_resource1
-encode_block_conv_add_readvariableop_resource8
4encode_block_1_conv_1_conv2d_readvariableop_resource5
1encode_block_1_conv_1_add_readvariableop_resource8
4encode_block_2_conv_2_conv2d_readvariableop_resource5
1encode_block_2_conv_2_add_readvariableop_resource8
4encode_block_3_conv_3_conv2d_readvariableop_resource5
1encode_block_3_conv_3_add_readvariableop_resource6
2decode_block_conv_4_conv2d_readvariableop_resource3
/decode_block_conv_4_add_readvariableop_resource8
4decode_block_1_conv_5_conv2d_readvariableop_resource5
1decode_block_1_conv_5_add_readvariableop_resource8
4decode_block_2_conv_6_conv2d_readvariableop_resource5
1decode_block_2_conv_6_add_readvariableop_resource8
4decode_block_3_conv_7_conv2d_readvariableop_resource5
1decode_block_3_conv_7_add_readvariableop_resource8
4decode_block_4_conv_8_conv2d_readvariableop_resource5
1decode_block_4_conv_8_add_readvariableop_resource
identityИҐ&decode_block/conv_4/Add/ReadVariableOpҐ)decode_block/conv_4/Conv2D/ReadVariableOpҐ(decode_block_1/conv_5/Add/ReadVariableOpҐ+decode_block_1/conv_5/Conv2D/ReadVariableOpҐ(decode_block_2/conv_6/Add/ReadVariableOpҐ+decode_block_2/conv_6/Conv2D/ReadVariableOpҐ(decode_block_3/conv_7/Add/ReadVariableOpҐ+decode_block_3/conv_7/Conv2D/ReadVariableOpҐ(decode_block_4/conv_8/Add/ReadVariableOpҐ+decode_block_4/conv_8/Conv2D/ReadVariableOpҐ$encode_block/conv/Add/ReadVariableOpҐ'encode_block/conv/Conv2D/ReadVariableOpҐ(encode_block_1/conv_1/Add/ReadVariableOpҐ+encode_block_1/conv_1/Conv2D/ReadVariableOpҐ(encode_block_2/conv_2/Add/ReadVariableOpҐ+encode_block_2/conv_2/Conv2D/ReadVariableOpҐ(encode_block_3/conv_3/Add/ReadVariableOpҐ+encode_block_3/conv_3/Conv2D/ReadVariableOpќ
'encode_block/conv/Conv2D/ReadVariableOpReadVariableOp0encode_block_conv_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@≈
encode_block/conv/Conv2DConv2Dinput_tensor/encode_block/conv/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іј@Љ
$encode_block/conv/Add/ReadVariableOpReadVariableOp-encode_block_conv_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@©
encode_block/conv/AddAdd!encode_block/conv/Conv2D:output:0,encode_block/conv/Add/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€іј@u
encode_block/conv/ReluReluencode_block/conv/Add:z:0*
T0*1
_output_shapes
:€€€€€€€€€іј@Ъ
,encode_block/conv/spatial_dropout2d/IdentityIdentity$encode_block/conv/Relu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€іј@Ы
encode_block/MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іјc
encode_block/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Ё
encode_block/concatConcatV25encode_block/conv/spatial_dropout2d/Identity:output:0encode_block/MaxPool:output:0!encode_block/concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€іјC÷
+encode_block_1/conv_1/Conv2D/ReadVariableOpReadVariableOp4encode_block_1_conv_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@х
encode_block_1/conv_1/Conv2DConv2D5encode_block/conv/spatial_dropout2d/Identity:output:03encode_block_1/conv_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@ƒ
(encode_block_1/conv_1/Add/ReadVariableOpReadVariableOp1encode_block_1_conv_1_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@і
encode_block_1/conv_1/AddAdd%encode_block_1/conv_1/Conv2D:output:00encode_block_1/conv_1/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@|
encode_block_1/conv_1/ReluReluencode_block_1/conv_1/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@£
2encode_block_1/conv_1/spatial_dropout2d_1/IdentityIdentity(encode_block_1/conv_1/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@≈
encode_block_1/MaxPoolMaxPool5encode_block/conv/spatial_dropout2d/Identity:output:0*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@e
encode_block_1/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: й
encode_block_1/concatConcatV2;encode_block_1/conv_1/spatial_dropout2d_1/Identity:output:0encode_block_1/MaxPool:output:0#encode_block_1/concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€Z†А÷
+encode_block_2/conv_2/Conv2D/ReadVariableOpReadVariableOp4encode_block_2_conv_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@ъ
encode_block_2/conv_2/Conv2DConv2D;encode_block_1/conv_1/spatial_dropout2d_1/Identity:output:03encode_block_2/conv_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@ƒ
(encode_block_2/conv_2/Add/ReadVariableOpReadVariableOp1encode_block_2_conv_2_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@≥
encode_block_2/conv_2/AddAdd%encode_block_2/conv_2/Conv2D:output:00encode_block_2/conv_2/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P@{
encode_block_2/conv_2/ReluReluencode_block_2/conv_2/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P@Ґ
2encode_block_2/conv_2/spatial_dropout2d_2/IdentityIdentity(encode_block_2/conv_2/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P@ 
encode_block_2/MaxPoolMaxPool;encode_block_1/conv_1/spatial_dropout2d_1/Identity:output:0*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@e
encode_block_2/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: и
encode_block_2/concatConcatV2;encode_block_2/conv_2/spatial_dropout2d_2/Identity:output:0encode_block_2/MaxPool:output:0#encode_block_2/concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€-PА÷
+encode_block_3/conv_3/Conv2D/ReadVariableOpReadVariableOp4encode_block_3_conv_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@ъ
encode_block_3/conv_3/Conv2DConv2D;encode_block_2/conv_2/spatial_dropout2d_2/Identity:output:03encode_block_3/conv_3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@ƒ
(encode_block_3/conv_3/Add/ReadVariableOpReadVariableOp1encode_block_3_conv_3_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@≥
encode_block_3/conv_3/AddAdd%encode_block_3/conv_3/Conv2D:output:00encode_block_3/conv_3/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€(@{
encode_block_3/conv_3/ReluReluencode_block_3/conv_3/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€(@Ґ
2encode_block_3/conv_3/spatial_dropout2d_3/IdentityIdentity(encode_block_3/conv_3/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€(@ 
encode_block_3/MaxPoolMaxPool;encode_block_2/conv_2/spatial_dropout2d_2/Identity:output:0*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@e
encode_block_3/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: и
encode_block_3/concatConcatV2;encode_block_3/conv_3/spatial_dropout2d_3/Identity:output:0encode_block_3/MaxPool:output:0#encode_block_3/concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€(А“
)decode_block/conv_4/Conv2D/ReadVariableOpReadVariableOp2decode_block_conv_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ц
decode_block/conv_4/Conv2DConv2D;encode_block_3/conv_3/spatial_dropout2d_3/Identity:output:01decode_block/conv_4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€( ј
&decode_block/conv_4/Add/ReadVariableOpReadVariableOp/decode_block_conv_4_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ≠
decode_block/conv_4/AddAdd#decode_block/conv_4/Conv2D:output:0.decode_block/conv_4/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€( w
decode_block/conv_4/ReluReludecode_block/conv_4/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€( Ю
0decode_block/conv_4/spatial_dropout2d_4/IdentityIdentity&decode_block/conv_4/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€( i
decode_block/resize/sizeConst*
valueB"-   P   *
dtype0*
_output_shapes
:ж
"decode_block/resize/ResizeBilinearResizeBilinear9decode_block/conv_4/spatial_dropout2d_4/Identity:output:0!decode_block/resize/size:output:0*
half_pixel_centers(*
T0*/
_output_shapes
:€€€€€€€€€-P ÷
+decode_block_1/conv_5/Conv2D/ReadVariableOpReadVariableOp4decode_block_1_conv_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  т
decode_block_1/conv_5/Conv2DConv2D3decode_block/resize/ResizeBilinear:resized_images:03decode_block_1/conv_5/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ƒ
(decode_block_1/conv_5/Add/ReadVariableOpReadVariableOp1decode_block_1_conv_5_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ≥
decode_block_1/conv_5/AddAdd%decode_block_1/conv_5/Conv2D:output:00decode_block_1/conv_5/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P {
decode_block_1/conv_5/ReluReludecode_block_1/conv_5/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P Ґ
2decode_block_1/conv_5/spatial_dropout2d_5/IdentityIdentity(decode_block_1/conv_5/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P k
decode_block_1/resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:н
$decode_block_1/resize/ResizeBilinearResizeBilinear;decode_block_1/conv_5/spatial_dropout2d_5/Identity:output:0#decode_block_1/resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ÷
+decode_block_2/conv_6/Conv2D/ReadVariableOpReadVariableOp4decode_block_2_conv_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ъ
decode_block_2/conv_6/Conv2DConv2D;encode_block_2/conv_2/spatial_dropout2d_2/Identity:output:03decode_block_2/conv_6/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ƒ
(decode_block_2/conv_6/Add/ReadVariableOpReadVariableOp1decode_block_2_conv_6_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ≥
decode_block_2/conv_6/AddAdd%decode_block_2/conv_6/Conv2D:output:00decode_block_2/conv_6/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P {
decode_block_2/conv_6/ReluReludecode_block_2/conv_6/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P Ґ
2decode_block_2/conv_6/spatial_dropout2d_6/IdentityIdentity(decode_block_2/conv_6/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P k
decode_block_2/resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:н
$decode_block_2/resize/ResizeBilinearResizeBilinear;decode_block_2/conv_6/spatial_dropout2d_6/Identity:output:0#decode_block_2/resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ÷
+decode_block_3/conv_7/Conv2D/ReadVariableOpReadVariableOp4decode_block_3_conv_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ы
decode_block_3/conv_7/Conv2DConv2D;encode_block_1/conv_1/spatial_dropout2d_1/Identity:output:03decode_block_3/conv_7/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z† ƒ
(decode_block_3/conv_7/Add/ReadVariableOpReadVariableOp1decode_block_3_conv_7_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: і
decode_block_3/conv_7/AddAdd%decode_block_3/conv_7/Conv2D:output:00decode_block_3/conv_7/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z† |
decode_block_3/conv_7/ReluReludecode_block_3/conv_7/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z† £
2decode_block_3/conv_7/spatial_dropout2d_7/IdentityIdentity(decode_block_3/conv_7/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€Z† k
decode_block_3/resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:н
$decode_block_3/resize/ResizeBilinearResizeBilinear;decode_block_3/conv_7/spatial_dropout2d_7/Identity:output:0#decode_block_3/resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: С
concatConcatV25decode_block_1/resize/ResizeBilinear:resized_images:05decode_block_2/resize/ResizeBilinear:resized_images:05decode_block_3/resize/ResizeBilinear:resized_images:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€Z†`÷
+decode_block_4/conv_8/Conv2D/ReadVariableOpReadVariableOp4decode_block_4_conv_8_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:`ѕ
decode_block_4/conv_8/Conv2DConv2Dconcat:output:03decode_block_4/conv_8/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†ƒ
(decode_block_4/conv_8/Add/ReadVariableOpReadVariableOp1decode_block_4_conv_8_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:і
decode_block_4/conv_8/AddAdd%decode_block_4/conv_8/Conv2D:output:00decode_block_4/conv_8/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†k
decode_block_4/resize/sizeConst*
valueB"h  А  *
dtype0*
_output_shapes
:–
$decode_block_4/resize/ResizeBilinearResizeBilineardecode_block_4/conv_8/Add:z:0#decode_block_4/resize/size:output:0*
half_pixel_centers(*
T0*1
_output_shapes
:€€€€€€€€€иАЕ
SigmoidSigmoid5decode_block_4/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:€€€€€€€€€иАт
IdentityIdentitySigmoid:y:0'^decode_block/conv_4/Add/ReadVariableOp*^decode_block/conv_4/Conv2D/ReadVariableOp)^decode_block_1/conv_5/Add/ReadVariableOp,^decode_block_1/conv_5/Conv2D/ReadVariableOp)^decode_block_2/conv_6/Add/ReadVariableOp,^decode_block_2/conv_6/Conv2D/ReadVariableOp)^decode_block_3/conv_7/Add/ReadVariableOp,^decode_block_3/conv_7/Conv2D/ReadVariableOp)^decode_block_4/conv_8/Add/ReadVariableOp,^decode_block_4/conv_8/Conv2D/ReadVariableOp%^encode_block/conv/Add/ReadVariableOp(^encode_block/conv/Conv2D/ReadVariableOp)^encode_block_1/conv_1/Add/ReadVariableOp,^encode_block_1/conv_1/Conv2D/ReadVariableOp)^encode_block_2/conv_2/Add/ReadVariableOp,^encode_block_2/conv_2/Conv2D/ReadVariableOp)^encode_block_3/conv_3/Add/ReadVariableOp,^encode_block_3/conv_3/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::2Z
+decode_block_4/conv_8/Conv2D/ReadVariableOp+decode_block_4/conv_8/Conv2D/ReadVariableOp2Z
+decode_block_3/conv_7/Conv2D/ReadVariableOp+decode_block_3/conv_7/Conv2D/ReadVariableOp2Z
+encode_block_3/conv_3/Conv2D/ReadVariableOp+encode_block_3/conv_3/Conv2D/ReadVariableOp2Z
+decode_block_2/conv_6/Conv2D/ReadVariableOp+decode_block_2/conv_6/Conv2D/ReadVariableOp2Z
+decode_block_1/conv_5/Conv2D/ReadVariableOp+decode_block_1/conv_5/Conv2D/ReadVariableOp2Z
+encode_block_2/conv_2/Conv2D/ReadVariableOp+encode_block_2/conv_2/Conv2D/ReadVariableOp2R
'encode_block/conv/Conv2D/ReadVariableOp'encode_block/conv/Conv2D/ReadVariableOp2Z
+encode_block_1/conv_1/Conv2D/ReadVariableOp+encode_block_1/conv_1/Conv2D/ReadVariableOp2T
(decode_block_2/conv_6/Add/ReadVariableOp(decode_block_2/conv_6/Add/ReadVariableOp2T
(encode_block_3/conv_3/Add/ReadVariableOp(encode_block_3/conv_3/Add/ReadVariableOp2V
)decode_block/conv_4/Conv2D/ReadVariableOp)decode_block/conv_4/Conv2D/ReadVariableOp2T
(encode_block_2/conv_2/Add/ReadVariableOp(encode_block_2/conv_2/Add/ReadVariableOp2T
(decode_block_1/conv_5/Add/ReadVariableOp(decode_block_1/conv_5/Add/ReadVariableOp2T
(decode_block_4/conv_8/Add/ReadVariableOp(decode_block_4/conv_8/Add/ReadVariableOp2P
&decode_block/conv_4/Add/ReadVariableOp&decode_block/conv_4/Add/ReadVariableOp2L
$encode_block/conv/Add/ReadVariableOp$encode_block/conv/Add/ReadVariableOp2T
(encode_block_1/conv_1/Add/ReadVariableOp(encode_block_1/conv_1/Add/ReadVariableOp2T
(decode_block_3/conv_7/Add/ReadVariableOp(decode_block_3/conv_7/Add/ReadVariableOp: : : :, (
&
_user_specified_nameinput_tensor: : : : :
 : : : : : : : : :	 : 
«
ь
I__inference_encode_block_1_layer_call_and_return_conditional_losses_12337
input_tensor)
%conv_1_conv2d_readvariableop_resource&
"conv_1_add_readvariableop_resource
identityИҐconv_1/Add/ReadVariableOpҐconv_1/Conv2D/ReadVariableOpЄ
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Ѓ
conv_1/Conv2DConv2Dinput_tensor$conv_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@¶
conv_1/Add/ReadVariableOpReadVariableOp"conv_1_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@З

conv_1/AddAddconv_1/Conv2D:output:0!conv_1/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@^
conv_1/ReluReluconv_1/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@i
 conv_1/spatial_dropout2d_1/ShapeShapeconv_1/Relu:activations:0*
T0*
_output_shapes
:x
.conv_1/spatial_dropout2d_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_1/spatial_dropout2d_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_1/spatial_dropout2d_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_1/spatial_dropout2d_1/strided_sliceStridedSlice)conv_1/spatial_dropout2d_1/Shape:output:07conv_1/spatial_dropout2d_1/strided_slice/stack:output:09conv_1/spatial_dropout2d_1/strided_slice/stack_1:output:09conv_1/spatial_dropout2d_1/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_1/spatial_dropout2d_1/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_1/spatial_dropout2d_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_1/spatial_dropout2d_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_1/spatial_dropout2d_1/strided_slice_1StridedSlice)conv_1/spatial_dropout2d_1/Shape:output:09conv_1/spatial_dropout2d_1/strided_slice_1/stack:output:0;conv_1/spatial_dropout2d_1/strided_slice_1/stack_1:output:0;conv_1/spatial_dropout2d_1/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Н
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Ъ
concatConcatV2conv_1/Relu:activations:0MaxPool:output:0concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€Z†А•
IdentityIdentityconv_1/Relu:activations:0^conv_1/Add/ReadVariableOp^conv_1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z†@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€іј@::2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp26
conv_1/Add/ReadVariableOpconv_1/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Т
µ
.__inference_decode_block_2_layer_call_fn_13656
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12677*R
fMRK
I__inference_decode_block_2_layer_call_and_return_conditional_losses_12663*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
Т
≥
,__inference_encode_block_layer_call_fn_13343
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12295*P
fKRI
G__inference_encode_block_layer_call_and_return_conditional_losses_12274*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€іј@М
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€іј@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€иА::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
Ј
O
3__inference_spatial_dropout2d_6_layer_call_fn_13940

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12190*W
fRRP
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_12189*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ч
l
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_12027

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
¶
j
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_13785

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ј
O
3__inference_spatial_dropout2d_4_layer_call_fn_13884

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12082*W
fRRP
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_12081*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
а
ь
I__inference_decode_block_4_layer_call_and_return_conditional_losses_13718
input_tensor)
%conv_8_conv2d_readvariableop_resource&
"conv_8_add_readvariableop_resource
identityИҐconv_8/Add/ReadVariableOpҐconv_8/Conv2D/ReadVariableOpЄ
conv_8/Conv2D/ReadVariableOpReadVariableOp%conv_8_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:`Ѓ
conv_8/Conv2DConv2Dinput_tensor$conv_8/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†¶
conv_8/Add/ReadVariableOpReadVariableOp"conv_8_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:З

conv_8/AddAddconv_8/Conv2D:output:0!conv_8/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†\
resize/sizeConst*
valueB"h  А  *
dtype0*
_output_shapes
:£
resize/ResizeBilinearResizeBilinearconv_8/Add:z:0resize/size:output:0*
half_pixel_centers(*
T0*1
_output_shapes
:€€€€€€€€€иА≥
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_8/Add/ReadVariableOp^conv_8/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†`::26
conv_8/Add/ReadVariableOpconv_8/Add/ReadVariableOp2<
conv_8/Conv2D/ReadVariableOpconv_8/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Ф
µ
.__inference_encode_block_1_layer_call_fn_13402
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12368*R
fMRK
I__inference_encode_block_1_layer_call_and_return_conditional_losses_12354*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z†@Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€Z†@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€іј@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
М
≥
,__inference_decode_block_layer_call_fn_13556
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12555*P
fKRI
G__inference_decode_block_layer_call_and_return_conditional_losses_12541*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P К
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€-P "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€(@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
а
ь
I__inference_decode_block_4_layer_call_and_return_conditional_losses_12775
input_tensor)
%conv_8_conv2d_readvariableop_resource&
"conv_8_add_readvariableop_resource
identityИҐconv_8/Add/ReadVariableOpҐconv_8/Conv2D/ReadVariableOpЄ
conv_8/Conv2D/ReadVariableOpReadVariableOp%conv_8_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:`Ѓ
conv_8/Conv2DConv2Dinput_tensor$conv_8/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†¶
conv_8/Add/ReadVariableOpReadVariableOp"conv_8_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:З

conv_8/AddAddconv_8/Conv2D:output:0!conv_8/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†\
resize/sizeConst*
valueB"h  А  *
dtype0*
_output_shapes
:£
resize/ResizeBilinearResizeBilinearconv_8/Add:z:0resize/size:output:0*
half_pixel_centers(*
T0*1
_output_shapes
:€€€€€€€€€иА≥
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_8/Add/ReadVariableOp^conv_8/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†`::26
conv_8/Add/ReadVariableOpconv_8/Add/ReadVariableOp2<
conv_8/Conv2D/ReadVariableOpconv_8/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
а
ь
I__inference_decode_block_4_layer_call_and_return_conditional_losses_13730
input_tensor)
%conv_8_conv2d_readvariableop_resource&
"conv_8_add_readvariableop_resource
identityИҐconv_8/Add/ReadVariableOpҐconv_8/Conv2D/ReadVariableOpЄ
conv_8/Conv2D/ReadVariableOpReadVariableOp%conv_8_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:`Ѓ
conv_8/Conv2DConv2Dinput_tensor$conv_8/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†¶
conv_8/Add/ReadVariableOpReadVariableOp"conv_8_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:З

conv_8/AddAddconv_8/Conv2D:output:0!conv_8/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†\
resize/sizeConst*
valueB"h  А  *
dtype0*
_output_shapes
:£
resize/ResizeBilinearResizeBilinearconv_8/Add:z:0resize/size:output:0*
half_pixel_centers(*
T0*1
_output_shapes
:€€€€€€€€€иА≥
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_8/Add/ReadVariableOp^conv_8/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†`::26
conv_8/Add/ReadVariableOpconv_8/Add/ReadVariableOp2<
conv_8/Conv2D/ReadVariableOpconv_8/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
ј:
џ

@__inference_model_layer_call_and_return_conditional_losses_12800
input_1/
+encode_block_statefulpartitionedcall_args_1/
+encode_block_statefulpartitionedcall_args_21
-encode_block_1_statefulpartitionedcall_args_11
-encode_block_1_statefulpartitionedcall_args_21
-encode_block_2_statefulpartitionedcall_args_11
-encode_block_2_statefulpartitionedcall_args_21
-encode_block_3_statefulpartitionedcall_args_11
-encode_block_3_statefulpartitionedcall_args_2/
+decode_block_statefulpartitionedcall_args_1/
+decode_block_statefulpartitionedcall_args_21
-decode_block_1_statefulpartitionedcall_args_11
-decode_block_1_statefulpartitionedcall_args_21
-decode_block_2_statefulpartitionedcall_args_11
-decode_block_2_statefulpartitionedcall_args_21
-decode_block_3_statefulpartitionedcall_args_11
-decode_block_3_statefulpartitionedcall_args_21
-decode_block_4_statefulpartitionedcall_args_11
-decode_block_4_statefulpartitionedcall_args_2
identityИҐ$decode_block/StatefulPartitionedCallҐ&decode_block_1/StatefulPartitionedCallҐ&decode_block_2/StatefulPartitionedCallҐ&decode_block_3/StatefulPartitionedCallҐ&decode_block_4/StatefulPartitionedCallҐ$encode_block/StatefulPartitionedCallҐ&encode_block_1/StatefulPartitionedCallҐ&encode_block_2/StatefulPartitionedCallҐ&encode_block_3/StatefulPartitionedCall•
$encode_block/StatefulPartitionedCallStatefulPartitionedCallinput_1+encode_block_statefulpartitionedcall_args_1+encode_block_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12295*P
fKRI
G__inference_encode_block_layer_call_and_return_conditional_losses_12274*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€іј@“
&encode_block_1/StatefulPartitionedCallStatefulPartitionedCall-encode_block/StatefulPartitionedCall:output:0-encode_block_1_statefulpartitionedcall_args_1-encode_block_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12358*R
fMRK
I__inference_encode_block_1_layer_call_and_return_conditional_losses_12337*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z†@”
&encode_block_2/StatefulPartitionedCallStatefulPartitionedCall/encode_block_1/StatefulPartitionedCall:output:0-encode_block_2_statefulpartitionedcall_args_1-encode_block_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12421*R
fMRK
I__inference_encode_block_2_layer_call_and_return_conditional_losses_12400*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P@”
&encode_block_3/StatefulPartitionedCallStatefulPartitionedCall/encode_block_2/StatefulPartitionedCall:output:0-encode_block_3_statefulpartitionedcall_args_1-encode_block_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12484*R
fMRK
I__inference_encode_block_3_layer_call_and_return_conditional_losses_12463*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€(@Ћ
$decode_block/StatefulPartitionedCallStatefulPartitionedCall/encode_block_3/StatefulPartitionedCall:output:0+decode_block_statefulpartitionedcall_args_1+decode_block_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12545*P
fKRI
G__inference_decode_block_layer_call_and_return_conditional_losses_12525*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P “
&decode_block_1/StatefulPartitionedCallStatefulPartitionedCall-decode_block/StatefulPartitionedCall:output:0-decode_block_1_statefulpartitionedcall_args_1-decode_block_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12606*R
fMRK
I__inference_decode_block_1_layer_call_and_return_conditional_losses_12586*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† ‘
&decode_block_2/StatefulPartitionedCallStatefulPartitionedCall/encode_block_2/StatefulPartitionedCall:output:0-decode_block_2_statefulpartitionedcall_args_1-decode_block_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12667*R
fMRK
I__inference_decode_block_2_layer_call_and_return_conditional_losses_12647*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† ‘
&decode_block_3/StatefulPartitionedCallStatefulPartitionedCall/encode_block_1/StatefulPartitionedCall:output:0-decode_block_3_statefulpartitionedcall_args_1-decode_block_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12728*R
fMRK
I__inference_decode_block_3_layer_call_and_return_conditional_losses_12708*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: €
concatConcatV2/decode_block_1/StatefulPartitionedCall:output:0/decode_block_2/StatefulPartitionedCall:output:0/decode_block_3/StatefulPartitionedCall:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€Z†`µ
&decode_block_4/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0-decode_block_4_statefulpartitionedcall_args_1-decode_block_4_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12779*R
fMRK
I__inference_decode_block_4_layer_call_and_return_conditional_losses_12761*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иА
SigmoidSigmoid/decode_block_4/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€иА 
IdentityIdentitySigmoid:y:0%^decode_block/StatefulPartitionedCall'^decode_block_1/StatefulPartitionedCall'^decode_block_2/StatefulPartitionedCall'^decode_block_3/StatefulPartitionedCall'^decode_block_4/StatefulPartitionedCall%^encode_block/StatefulPartitionedCall'^encode_block_1/StatefulPartitionedCall'^encode_block_2/StatefulPartitionedCall'^encode_block_3/StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::2P
&encode_block_1/StatefulPartitionedCall&encode_block_1/StatefulPartitionedCall2P
&encode_block_2/StatefulPartitionedCall&encode_block_2/StatefulPartitionedCall2P
&decode_block_1/StatefulPartitionedCall&decode_block_1/StatefulPartitionedCall2P
&encode_block_3/StatefulPartitionedCall&encode_block_3/StatefulPartitionedCall2P
&decode_block_2/StatefulPartitionedCall&decode_block_2/StatefulPartitionedCall2P
&decode_block_3/StatefulPartitionedCall&decode_block_3/StatefulPartitionedCall2L
$encode_block/StatefulPartitionedCall$encode_block/StatefulPartitionedCall2P
&decode_block_4/StatefulPartitionedCall&decode_block_4/StatefulPartitionedCall2L
$decode_block/StatefulPartitionedCall$decode_block/StatefulPartitionedCall: : : :' #
!
_user_specified_name	input_1: : : : :
 : : : : : : : : :	 : 
≥
M
1__inference_spatial_dropout2d_layer_call_fn_13767

inputs
identity…
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-11853*U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_11852*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ј
O
3__inference_spatial_dropout2d_5_layer_call_fn_13912

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12136*W
fRRP
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_12135*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Р
µ
.__inference_encode_block_3_layer_call_fn_13499
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12484*R
fMRK
I__inference_encode_block_3_layer_call_and_return_conditional_losses_12463*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€(@К
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€(@"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
Ј
O
3__inference_spatial_dropout2d_4_layer_call_fn_13879

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12069*W
fRRP
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_12068*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
њ
ь
I__inference_encode_block_3_layer_call_and_return_conditional_losses_12463
input_tensor)
%conv_3_conv2d_readvariableop_resource&
"conv_3_add_readvariableop_resource
identityИҐconv_3/Add/ReadVariableOpҐconv_3/Conv2D/ReadVariableOpЄ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@≠
conv_3/Conv2DConv2Dinput_tensor$conv_3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@¶
conv_3/Add/ReadVariableOpReadVariableOp"conv_3_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ж

conv_3/AddAddconv_3/Conv2D:output:0!conv_3/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€(@]
conv_3/ReluReluconv_3/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€(@i
 conv_3/spatial_dropout2d_3/ShapeShapeconv_3/Relu:activations:0*
T0*
_output_shapes
:x
.conv_3/spatial_dropout2d_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_3/spatial_dropout2d_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_3/spatial_dropout2d_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_3/spatial_dropout2d_3/strided_sliceStridedSlice)conv_3/spatial_dropout2d_3/Shape:output:07conv_3/spatial_dropout2d_3/strided_slice/stack:output:09conv_3/spatial_dropout2d_3/strided_slice/stack_1:output:09conv_3/spatial_dropout2d_3/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_3/spatial_dropout2d_3/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_3/spatial_dropout2d_3/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_3/spatial_dropout2d_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_3/spatial_dropout2d_3/strided_slice_1StridedSlice)conv_3/spatial_dropout2d_3/Shape:output:09conv_3/spatial_dropout2d_3/strided_slice_1/stack:output:0;conv_3/spatial_dropout2d_3/strided_slice_1/stack_1:output:0;conv_3/spatial_dropout2d_3/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: М
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Щ
concatConcatV2conv_3/Relu:activations:0MaxPool:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€(А§
IdentityIdentityconv_3/Relu:activations:0^conv_3/Add/ReadVariableOp^conv_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€(@"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::26
conv_3/Add/ReadVariableOpconv_3/Add/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Ј
O
3__inference_spatial_dropout2d_3_layer_call_fn_13851

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12015*W
fRRP
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_12014*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
¶
j
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_13869

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ж
х
%__inference_model_layer_call_fn_13275
input_tensor"
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
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*,
_gradient_op_typePartitionedCall-12874*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_12873*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иАМ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :, (
&
_user_specified_nameinput_tensor: : : : :
 : : : : : : : : :	 : 
ч
l
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_13902

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Т
≥
,__inference_encode_block_layer_call_fn_13350
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12305*P
fKRI
G__inference_encode_block_layer_call_and_return_conditional_losses_12291*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€іј@М
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€іј@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€иА::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
¶
j
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_13813

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
‘
ъ
G__inference_decode_block_layer_call_and_return_conditional_losses_12525
input_tensor)
%conv_4_conv2d_readvariableop_resource&
"conv_4_add_readvariableop_resource
identityИҐconv_4/Add/ReadVariableOpҐconv_4/Conv2D/ReadVariableOpЄ
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ≠
conv_4/Conv2DConv2Dinput_tensor$conv_4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€( ¶
conv_4/Add/ReadVariableOpReadVariableOp"conv_4_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_4/AddAddconv_4/Conv2D:output:0!conv_4/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€( ]
conv_4/ReluReluconv_4/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€( i
 conv_4/spatial_dropout2d_4/ShapeShapeconv_4/Relu:activations:0*
T0*
_output_shapes
:x
.conv_4/spatial_dropout2d_4/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_4/spatial_dropout2d_4/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_4/spatial_dropout2d_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_4/spatial_dropout2d_4/strided_sliceStridedSlice)conv_4/spatial_dropout2d_4/Shape:output:07conv_4/spatial_dropout2d_4/strided_slice/stack:output:09conv_4/spatial_dropout2d_4/strided_slice/stack_1:output:09conv_4/spatial_dropout2d_4/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_4/spatial_dropout2d_4/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_4/spatial_dropout2d_4/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_4/spatial_dropout2d_4/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_4/spatial_dropout2d_4/strided_slice_1StridedSlice)conv_4/spatial_dropout2d_4/Shape:output:09conv_4/spatial_dropout2d_4/strided_slice_1/stack:output:0;conv_4/spatial_dropout2d_4/strided_slice_1/stack_1:output:0;conv_4/spatial_dropout2d_4/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: \
resize/sizeConst*
valueB"-   P   *
dtype0*
_output_shapes
:ђ
resize/ResizeBilinearResizeBilinearconv_4/Relu:activations:0resize/size:output:0*
half_pixel_centers(*
T0*/
_output_shapes
:€€€€€€€€€-P ±
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_4/Add/ReadVariableOp^conv_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€-P "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€(@::2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp26
conv_4/Add/ReadVariableOpconv_4/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
¶
j
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_12176

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ѕч
з
@__inference_model_layer_call_and_return_conditional_losses_13153
input_tensor4
0encode_block_conv_conv2d_readvariableop_resource1
-encode_block_conv_add_readvariableop_resource8
4encode_block_1_conv_1_conv2d_readvariableop_resource5
1encode_block_1_conv_1_add_readvariableop_resource8
4encode_block_2_conv_2_conv2d_readvariableop_resource5
1encode_block_2_conv_2_add_readvariableop_resource8
4encode_block_3_conv_3_conv2d_readvariableop_resource5
1encode_block_3_conv_3_add_readvariableop_resource6
2decode_block_conv_4_conv2d_readvariableop_resource3
/decode_block_conv_4_add_readvariableop_resource8
4decode_block_1_conv_5_conv2d_readvariableop_resource5
1decode_block_1_conv_5_add_readvariableop_resource8
4decode_block_2_conv_6_conv2d_readvariableop_resource5
1decode_block_2_conv_6_add_readvariableop_resource8
4decode_block_3_conv_7_conv2d_readvariableop_resource5
1decode_block_3_conv_7_add_readvariableop_resource8
4decode_block_4_conv_8_conv2d_readvariableop_resource5
1decode_block_4_conv_8_add_readvariableop_resource
identityИҐ&decode_block/conv_4/Add/ReadVariableOpҐ)decode_block/conv_4/Conv2D/ReadVariableOpҐ(decode_block_1/conv_5/Add/ReadVariableOpҐ+decode_block_1/conv_5/Conv2D/ReadVariableOpҐ(decode_block_2/conv_6/Add/ReadVariableOpҐ+decode_block_2/conv_6/Conv2D/ReadVariableOpҐ(decode_block_3/conv_7/Add/ReadVariableOpҐ+decode_block_3/conv_7/Conv2D/ReadVariableOpҐ(decode_block_4/conv_8/Add/ReadVariableOpҐ+decode_block_4/conv_8/Conv2D/ReadVariableOpҐ$encode_block/conv/Add/ReadVariableOpҐ'encode_block/conv/Conv2D/ReadVariableOpҐ(encode_block_1/conv_1/Add/ReadVariableOpҐ+encode_block_1/conv_1/Conv2D/ReadVariableOpҐ(encode_block_2/conv_2/Add/ReadVariableOpҐ+encode_block_2/conv_2/Conv2D/ReadVariableOpҐ(encode_block_3/conv_3/Add/ReadVariableOpҐ+encode_block_3/conv_3/Conv2D/ReadVariableOpќ
'encode_block/conv/Conv2D/ReadVariableOpReadVariableOp0encode_block_conv_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@≈
encode_block/conv/Conv2DConv2Dinput_tensor/encode_block/conv/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іј@Љ
$encode_block/conv/Add/ReadVariableOpReadVariableOp-encode_block_conv_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@©
encode_block/conv/AddAdd!encode_block/conv/Conv2D:output:0,encode_block/conv/Add/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€іј@u
encode_block/conv/ReluReluencode_block/conv/Add:z:0*
T0*1
_output_shapes
:€€€€€€€€€іј@}
)encode_block/conv/spatial_dropout2d/ShapeShape$encode_block/conv/Relu:activations:0*
T0*
_output_shapes
:Б
7encode_block/conv/spatial_dropout2d/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:Г
9encode_block/conv/spatial_dropout2d/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Г
9encode_block/conv/spatial_dropout2d/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Е
1encode_block/conv/spatial_dropout2d/strided_sliceStridedSlice2encode_block/conv/spatial_dropout2d/Shape:output:0@encode_block/conv/spatial_dropout2d/strided_slice/stack:output:0Bencode_block/conv/spatial_dropout2d/strided_slice/stack_1:output:0Bencode_block/conv/spatial_dropout2d/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Г
9encode_block/conv/spatial_dropout2d/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:Е
;encode_block/conv/spatial_dropout2d/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Е
;encode_block/conv/spatial_dropout2d/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Н
3encode_block/conv/spatial_dropout2d/strided_slice_1StridedSlice2encode_block/conv/spatial_dropout2d/Shape:output:0Bencode_block/conv/spatial_dropout2d/strided_slice_1/stack:output:0Dencode_block/conv/spatial_dropout2d/strided_slice_1/stack_1:output:0Dencode_block/conv/spatial_dropout2d/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Ы
encode_block/MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іјc
encode_block/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ћ
encode_block/concatConcatV2$encode_block/conv/Relu:activations:0encode_block/MaxPool:output:0!encode_block/concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€іјC÷
+encode_block_1/conv_1/Conv2D/ReadVariableOpReadVariableOp4encode_block_1_conv_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@д
encode_block_1/conv_1/Conv2DConv2D$encode_block/conv/Relu:activations:03encode_block_1/conv_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@ƒ
(encode_block_1/conv_1/Add/ReadVariableOpReadVariableOp1encode_block_1_conv_1_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@і
encode_block_1/conv_1/AddAdd%encode_block_1/conv_1/Conv2D:output:00encode_block_1/conv_1/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@|
encode_block_1/conv_1/ReluReluencode_block_1/conv_1/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@З
/encode_block_1/conv_1/spatial_dropout2d_1/ShapeShape(encode_block_1/conv_1/Relu:activations:0*
T0*
_output_shapes
:З
=encode_block_1/conv_1/spatial_dropout2d_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:Й
?encode_block_1/conv_1/spatial_dropout2d_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Й
?encode_block_1/conv_1/spatial_dropout2d_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:£
7encode_block_1/conv_1/spatial_dropout2d_1/strided_sliceStridedSlice8encode_block_1/conv_1/spatial_dropout2d_1/Shape:output:0Fencode_block_1/conv_1/spatial_dropout2d_1/strided_slice/stack:output:0Hencode_block_1/conv_1/spatial_dropout2d_1/strided_slice/stack_1:output:0Hencode_block_1/conv_1/spatial_dropout2d_1/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Й
?encode_block_1/conv_1/spatial_dropout2d_1/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:Л
Aencode_block_1/conv_1/spatial_dropout2d_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Л
Aencode_block_1/conv_1/spatial_dropout2d_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ђ
9encode_block_1/conv_1/spatial_dropout2d_1/strided_slice_1StridedSlice8encode_block_1/conv_1/spatial_dropout2d_1/Shape:output:0Hencode_block_1/conv_1/spatial_dropout2d_1/strided_slice_1/stack:output:0Jencode_block_1/conv_1/spatial_dropout2d_1/strided_slice_1/stack_1:output:0Jencode_block_1/conv_1/spatial_dropout2d_1/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: і
encode_block_1/MaxPoolMaxPool$encode_block/conv/Relu:activations:0*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@e
encode_block_1/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ÷
encode_block_1/concatConcatV2(encode_block_1/conv_1/Relu:activations:0encode_block_1/MaxPool:output:0#encode_block_1/concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€Z†А÷
+encode_block_2/conv_2/Conv2D/ReadVariableOpReadVariableOp4encode_block_2_conv_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@з
encode_block_2/conv_2/Conv2DConv2D(encode_block_1/conv_1/Relu:activations:03encode_block_2/conv_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@ƒ
(encode_block_2/conv_2/Add/ReadVariableOpReadVariableOp1encode_block_2_conv_2_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@≥
encode_block_2/conv_2/AddAdd%encode_block_2/conv_2/Conv2D:output:00encode_block_2/conv_2/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P@{
encode_block_2/conv_2/ReluReluencode_block_2/conv_2/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P@З
/encode_block_2/conv_2/spatial_dropout2d_2/ShapeShape(encode_block_2/conv_2/Relu:activations:0*
T0*
_output_shapes
:З
=encode_block_2/conv_2/spatial_dropout2d_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:Й
?encode_block_2/conv_2/spatial_dropout2d_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Й
?encode_block_2/conv_2/spatial_dropout2d_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:£
7encode_block_2/conv_2/spatial_dropout2d_2/strided_sliceStridedSlice8encode_block_2/conv_2/spatial_dropout2d_2/Shape:output:0Fencode_block_2/conv_2/spatial_dropout2d_2/strided_slice/stack:output:0Hencode_block_2/conv_2/spatial_dropout2d_2/strided_slice/stack_1:output:0Hencode_block_2/conv_2/spatial_dropout2d_2/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Й
?encode_block_2/conv_2/spatial_dropout2d_2/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:Л
Aencode_block_2/conv_2/spatial_dropout2d_2/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Л
Aencode_block_2/conv_2/spatial_dropout2d_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ђ
9encode_block_2/conv_2/spatial_dropout2d_2/strided_slice_1StridedSlice8encode_block_2/conv_2/spatial_dropout2d_2/Shape:output:0Hencode_block_2/conv_2/spatial_dropout2d_2/strided_slice_1/stack:output:0Jencode_block_2/conv_2/spatial_dropout2d_2/strided_slice_1/stack_1:output:0Jencode_block_2/conv_2/spatial_dropout2d_2/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Ј
encode_block_2/MaxPoolMaxPool(encode_block_1/conv_1/Relu:activations:0*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@e
encode_block_2/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ’
encode_block_2/concatConcatV2(encode_block_2/conv_2/Relu:activations:0encode_block_2/MaxPool:output:0#encode_block_2/concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€-PА÷
+encode_block_3/conv_3/Conv2D/ReadVariableOpReadVariableOp4encode_block_3_conv_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@з
encode_block_3/conv_3/Conv2DConv2D(encode_block_2/conv_2/Relu:activations:03encode_block_3/conv_3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@ƒ
(encode_block_3/conv_3/Add/ReadVariableOpReadVariableOp1encode_block_3_conv_3_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@≥
encode_block_3/conv_3/AddAdd%encode_block_3/conv_3/Conv2D:output:00encode_block_3/conv_3/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€(@{
encode_block_3/conv_3/ReluReluencode_block_3/conv_3/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€(@З
/encode_block_3/conv_3/spatial_dropout2d_3/ShapeShape(encode_block_3/conv_3/Relu:activations:0*
T0*
_output_shapes
:З
=encode_block_3/conv_3/spatial_dropout2d_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:Й
?encode_block_3/conv_3/spatial_dropout2d_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Й
?encode_block_3/conv_3/spatial_dropout2d_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:£
7encode_block_3/conv_3/spatial_dropout2d_3/strided_sliceStridedSlice8encode_block_3/conv_3/spatial_dropout2d_3/Shape:output:0Fencode_block_3/conv_3/spatial_dropout2d_3/strided_slice/stack:output:0Hencode_block_3/conv_3/spatial_dropout2d_3/strided_slice/stack_1:output:0Hencode_block_3/conv_3/spatial_dropout2d_3/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Й
?encode_block_3/conv_3/spatial_dropout2d_3/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:Л
Aencode_block_3/conv_3/spatial_dropout2d_3/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Л
Aencode_block_3/conv_3/spatial_dropout2d_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ђ
9encode_block_3/conv_3/spatial_dropout2d_3/strided_slice_1StridedSlice8encode_block_3/conv_3/spatial_dropout2d_3/Shape:output:0Hencode_block_3/conv_3/spatial_dropout2d_3/strided_slice_1/stack:output:0Jencode_block_3/conv_3/spatial_dropout2d_3/strided_slice_1/stack_1:output:0Jencode_block_3/conv_3/spatial_dropout2d_3/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Ј
encode_block_3/MaxPoolMaxPool(encode_block_2/conv_2/Relu:activations:0*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@e
encode_block_3/concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ’
encode_block_3/concatConcatV2(encode_block_3/conv_3/Relu:activations:0encode_block_3/MaxPool:output:0#encode_block_3/concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€(А“
)decode_block/conv_4/Conv2D/ReadVariableOpReadVariableOp2decode_block_conv_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ г
decode_block/conv_4/Conv2DConv2D(encode_block_3/conv_3/Relu:activations:01decode_block/conv_4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€( ј
&decode_block/conv_4/Add/ReadVariableOpReadVariableOp/decode_block_conv_4_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ≠
decode_block/conv_4/AddAdd#decode_block/conv_4/Conv2D:output:0.decode_block/conv_4/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€( w
decode_block/conv_4/ReluReludecode_block/conv_4/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€( Г
-decode_block/conv_4/spatial_dropout2d_4/ShapeShape&decode_block/conv_4/Relu:activations:0*
T0*
_output_shapes
:Е
;decode_block/conv_4/spatial_dropout2d_4/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:З
=decode_block/conv_4/spatial_dropout2d_4/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:З
=decode_block/conv_4/spatial_dropout2d_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Щ
5decode_block/conv_4/spatial_dropout2d_4/strided_sliceStridedSlice6decode_block/conv_4/spatial_dropout2d_4/Shape:output:0Ddecode_block/conv_4/spatial_dropout2d_4/strided_slice/stack:output:0Fdecode_block/conv_4/spatial_dropout2d_4/strided_slice/stack_1:output:0Fdecode_block/conv_4/spatial_dropout2d_4/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: З
=decode_block/conv_4/spatial_dropout2d_4/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:Й
?decode_block/conv_4/spatial_dropout2d_4/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Й
?decode_block/conv_4/spatial_dropout2d_4/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:°
7decode_block/conv_4/spatial_dropout2d_4/strided_slice_1StridedSlice6decode_block/conv_4/spatial_dropout2d_4/Shape:output:0Fdecode_block/conv_4/spatial_dropout2d_4/strided_slice_1/stack:output:0Hdecode_block/conv_4/spatial_dropout2d_4/strided_slice_1/stack_1:output:0Hdecode_block/conv_4/spatial_dropout2d_4/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: i
decode_block/resize/sizeConst*
valueB"-   P   *
dtype0*
_output_shapes
:”
"decode_block/resize/ResizeBilinearResizeBilinear&decode_block/conv_4/Relu:activations:0!decode_block/resize/size:output:0*
half_pixel_centers(*
T0*/
_output_shapes
:€€€€€€€€€-P ÷
+decode_block_1/conv_5/Conv2D/ReadVariableOpReadVariableOp4decode_block_1_conv_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  т
decode_block_1/conv_5/Conv2DConv2D3decode_block/resize/ResizeBilinear:resized_images:03decode_block_1/conv_5/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ƒ
(decode_block_1/conv_5/Add/ReadVariableOpReadVariableOp1decode_block_1_conv_5_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ≥
decode_block_1/conv_5/AddAdd%decode_block_1/conv_5/Conv2D:output:00decode_block_1/conv_5/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P {
decode_block_1/conv_5/ReluReludecode_block_1/conv_5/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P З
/decode_block_1/conv_5/spatial_dropout2d_5/ShapeShape(decode_block_1/conv_5/Relu:activations:0*
T0*
_output_shapes
:З
=decode_block_1/conv_5/spatial_dropout2d_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:Й
?decode_block_1/conv_5/spatial_dropout2d_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Й
?decode_block_1/conv_5/spatial_dropout2d_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:£
7decode_block_1/conv_5/spatial_dropout2d_5/strided_sliceStridedSlice8decode_block_1/conv_5/spatial_dropout2d_5/Shape:output:0Fdecode_block_1/conv_5/spatial_dropout2d_5/strided_slice/stack:output:0Hdecode_block_1/conv_5/spatial_dropout2d_5/strided_slice/stack_1:output:0Hdecode_block_1/conv_5/spatial_dropout2d_5/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Й
?decode_block_1/conv_5/spatial_dropout2d_5/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:Л
Adecode_block_1/conv_5/spatial_dropout2d_5/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Л
Adecode_block_1/conv_5/spatial_dropout2d_5/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ђ
9decode_block_1/conv_5/spatial_dropout2d_5/strided_slice_1StridedSlice8decode_block_1/conv_5/spatial_dropout2d_5/Shape:output:0Hdecode_block_1/conv_5/spatial_dropout2d_5/strided_slice_1/stack:output:0Jdecode_block_1/conv_5/spatial_dropout2d_5/strided_slice_1/stack_1:output:0Jdecode_block_1/conv_5/spatial_dropout2d_5/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: k
decode_block_1/resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:Џ
$decode_block_1/resize/ResizeBilinearResizeBilinear(decode_block_1/conv_5/Relu:activations:0#decode_block_1/resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ÷
+decode_block_2/conv_6/Conv2D/ReadVariableOpReadVariableOp4decode_block_2_conv_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ з
decode_block_2/conv_6/Conv2DConv2D(encode_block_2/conv_2/Relu:activations:03decode_block_2/conv_6/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ƒ
(decode_block_2/conv_6/Add/ReadVariableOpReadVariableOp1decode_block_2_conv_6_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ≥
decode_block_2/conv_6/AddAdd%decode_block_2/conv_6/Conv2D:output:00decode_block_2/conv_6/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P {
decode_block_2/conv_6/ReluReludecode_block_2/conv_6/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P З
/decode_block_2/conv_6/spatial_dropout2d_6/ShapeShape(decode_block_2/conv_6/Relu:activations:0*
T0*
_output_shapes
:З
=decode_block_2/conv_6/spatial_dropout2d_6/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:Й
?decode_block_2/conv_6/spatial_dropout2d_6/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Й
?decode_block_2/conv_6/spatial_dropout2d_6/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:£
7decode_block_2/conv_6/spatial_dropout2d_6/strided_sliceStridedSlice8decode_block_2/conv_6/spatial_dropout2d_6/Shape:output:0Fdecode_block_2/conv_6/spatial_dropout2d_6/strided_slice/stack:output:0Hdecode_block_2/conv_6/spatial_dropout2d_6/strided_slice/stack_1:output:0Hdecode_block_2/conv_6/spatial_dropout2d_6/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Й
?decode_block_2/conv_6/spatial_dropout2d_6/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:Л
Adecode_block_2/conv_6/spatial_dropout2d_6/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Л
Adecode_block_2/conv_6/spatial_dropout2d_6/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ђ
9decode_block_2/conv_6/spatial_dropout2d_6/strided_slice_1StridedSlice8decode_block_2/conv_6/spatial_dropout2d_6/Shape:output:0Hdecode_block_2/conv_6/spatial_dropout2d_6/strided_slice_1/stack:output:0Jdecode_block_2/conv_6/spatial_dropout2d_6/strided_slice_1/stack_1:output:0Jdecode_block_2/conv_6/spatial_dropout2d_6/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: k
decode_block_2/resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:Џ
$decode_block_2/resize/ResizeBilinearResizeBilinear(decode_block_2/conv_6/Relu:activations:0#decode_block_2/resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ÷
+decode_block_3/conv_7/Conv2D/ReadVariableOpReadVariableOp4decode_block_3_conv_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ и
decode_block_3/conv_7/Conv2DConv2D(encode_block_1/conv_1/Relu:activations:03decode_block_3/conv_7/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z† ƒ
(decode_block_3/conv_7/Add/ReadVariableOpReadVariableOp1decode_block_3_conv_7_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: і
decode_block_3/conv_7/AddAdd%decode_block_3/conv_7/Conv2D:output:00decode_block_3/conv_7/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z† |
decode_block_3/conv_7/ReluReludecode_block_3/conv_7/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z† З
/decode_block_3/conv_7/spatial_dropout2d_7/ShapeShape(decode_block_3/conv_7/Relu:activations:0*
T0*
_output_shapes
:З
=decode_block_3/conv_7/spatial_dropout2d_7/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:Й
?decode_block_3/conv_7/spatial_dropout2d_7/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Й
?decode_block_3/conv_7/spatial_dropout2d_7/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:£
7decode_block_3/conv_7/spatial_dropout2d_7/strided_sliceStridedSlice8decode_block_3/conv_7/spatial_dropout2d_7/Shape:output:0Fdecode_block_3/conv_7/spatial_dropout2d_7/strided_slice/stack:output:0Hdecode_block_3/conv_7/spatial_dropout2d_7/strided_slice/stack_1:output:0Hdecode_block_3/conv_7/spatial_dropout2d_7/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Й
?decode_block_3/conv_7/spatial_dropout2d_7/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:Л
Adecode_block_3/conv_7/spatial_dropout2d_7/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Л
Adecode_block_3/conv_7/spatial_dropout2d_7/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ђ
9decode_block_3/conv_7/spatial_dropout2d_7/strided_slice_1StridedSlice8decode_block_3/conv_7/spatial_dropout2d_7/Shape:output:0Hdecode_block_3/conv_7/spatial_dropout2d_7/strided_slice_1/stack:output:0Jdecode_block_3/conv_7/spatial_dropout2d_7/strided_slice_1/stack_1:output:0Jdecode_block_3/conv_7/spatial_dropout2d_7/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: k
decode_block_3/resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:Џ
$decode_block_3/resize/ResizeBilinearResizeBilinear(decode_block_3/conv_7/Relu:activations:0#decode_block_3/resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: С
concatConcatV25decode_block_1/resize/ResizeBilinear:resized_images:05decode_block_2/resize/ResizeBilinear:resized_images:05decode_block_3/resize/ResizeBilinear:resized_images:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€Z†`÷
+decode_block_4/conv_8/Conv2D/ReadVariableOpReadVariableOp4decode_block_4_conv_8_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:`ѕ
decode_block_4/conv_8/Conv2DConv2Dconcat:output:03decode_block_4/conv_8/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†ƒ
(decode_block_4/conv_8/Add/ReadVariableOpReadVariableOp1decode_block_4_conv_8_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:і
decode_block_4/conv_8/AddAdd%decode_block_4/conv_8/Conv2D:output:00decode_block_4/conv_8/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†k
decode_block_4/resize/sizeConst*
valueB"h  А  *
dtype0*
_output_shapes
:–
$decode_block_4/resize/ResizeBilinearResizeBilineardecode_block_4/conv_8/Add:z:0#decode_block_4/resize/size:output:0*
half_pixel_centers(*
T0*1
_output_shapes
:€€€€€€€€€иАЕ
SigmoidSigmoid5decode_block_4/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:€€€€€€€€€иАт
IdentityIdentitySigmoid:y:0'^decode_block/conv_4/Add/ReadVariableOp*^decode_block/conv_4/Conv2D/ReadVariableOp)^decode_block_1/conv_5/Add/ReadVariableOp,^decode_block_1/conv_5/Conv2D/ReadVariableOp)^decode_block_2/conv_6/Add/ReadVariableOp,^decode_block_2/conv_6/Conv2D/ReadVariableOp)^decode_block_3/conv_7/Add/ReadVariableOp,^decode_block_3/conv_7/Conv2D/ReadVariableOp)^decode_block_4/conv_8/Add/ReadVariableOp,^decode_block_4/conv_8/Conv2D/ReadVariableOp%^encode_block/conv/Add/ReadVariableOp(^encode_block/conv/Conv2D/ReadVariableOp)^encode_block_1/conv_1/Add/ReadVariableOp,^encode_block_1/conv_1/Conv2D/ReadVariableOp)^encode_block_2/conv_2/Add/ReadVariableOp,^encode_block_2/conv_2/Conv2D/ReadVariableOp)^encode_block_3/conv_3/Add/ReadVariableOp,^encode_block_3/conv_3/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::2Z
+decode_block_4/conv_8/Conv2D/ReadVariableOp+decode_block_4/conv_8/Conv2D/ReadVariableOp2Z
+decode_block_3/conv_7/Conv2D/ReadVariableOp+decode_block_3/conv_7/Conv2D/ReadVariableOp2Z
+encode_block_3/conv_3/Conv2D/ReadVariableOp+encode_block_3/conv_3/Conv2D/ReadVariableOp2Z
+decode_block_2/conv_6/Conv2D/ReadVariableOp+decode_block_2/conv_6/Conv2D/ReadVariableOp2Z
+encode_block_2/conv_2/Conv2D/ReadVariableOp+encode_block_2/conv_2/Conv2D/ReadVariableOp2Z
+decode_block_1/conv_5/Conv2D/ReadVariableOp+decode_block_1/conv_5/Conv2D/ReadVariableOp2R
'encode_block/conv/Conv2D/ReadVariableOp'encode_block/conv/Conv2D/ReadVariableOp2Z
+encode_block_1/conv_1/Conv2D/ReadVariableOp+encode_block_1/conv_1/Conv2D/ReadVariableOp2T
(encode_block_3/conv_3/Add/ReadVariableOp(encode_block_3/conv_3/Add/ReadVariableOp2T
(decode_block_2/conv_6/Add/ReadVariableOp(decode_block_2/conv_6/Add/ReadVariableOp2V
)decode_block/conv_4/Conv2D/ReadVariableOp)decode_block/conv_4/Conv2D/ReadVariableOp2T
(decode_block_1/conv_5/Add/ReadVariableOp(decode_block_1/conv_5/Add/ReadVariableOp2T
(encode_block_2/conv_2/Add/ReadVariableOp(encode_block_2/conv_2/Add/ReadVariableOp2T
(decode_block_4/conv_8/Add/ReadVariableOp(decode_block_4/conv_8/Add/ReadVariableOp2P
&decode_block/conv_4/Add/ReadVariableOp&decode_block/conv_4/Add/ReadVariableOp2L
$encode_block/conv/Add/ReadVariableOp$encode_block/conv/Add/ReadVariableOp2T
(encode_block_1/conv_1/Add/ReadVariableOp(encode_block_1/conv_1/Add/ReadVariableOp2T
(decode_block_3/conv_7/Add/ReadVariableOp(decode_block_3/conv_7/Add/ReadVariableOp: : : :, (
&
_user_specified_nameinput_tensor: : : : :
 : : : : : : : : :	 : 
¶
j
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_11906

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ч
l
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_12189

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ч
l
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_11973

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
д
ь
I__inference_decode_block_3_layer_call_and_return_conditional_losses_13692
input_tensor)
%conv_7_conv2d_readvariableop_resource&
"conv_7_add_readvariableop_resource
identityИҐconv_7/Add/ReadVariableOpҐconv_7/Conv2D/ReadVariableOpЄ
conv_7/Conv2D/ReadVariableOpReadVariableOp%conv_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ Ѓ
conv_7/Conv2DConv2Dinput_tensor$conv_7/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z† ¶
conv_7/Add/ReadVariableOpReadVariableOp"conv_7_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: З

conv_7/AddAddconv_7/Conv2D:output:0!conv_7/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z† ^
conv_7/ReluReluconv_7/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z† Е
#conv_7/spatial_dropout2d_7/IdentityIdentityconv_7/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€Z† \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:ј
resize/ResizeBilinearResizeBilinear,conv_7/spatial_dropout2d_7/Identity:output:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_7/Add/ReadVariableOp^conv_7/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::2<
conv_7/Conv2D/ReadVariableOpconv_7/Conv2D/ReadVariableOp26
conv_7/Add/ReadVariableOpconv_7/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
ч
l
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_11919

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ј
ь
I__inference_encode_block_2_layer_call_and_return_conditional_losses_12400
input_tensor)
%conv_2_conv2d_readvariableop_resource&
"conv_2_add_readvariableop_resource
identityИҐconv_2/Add/ReadVariableOpҐconv_2/Conv2D/ReadVariableOpЄ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@≠
conv_2/Conv2DConv2Dinput_tensor$conv_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@¶
conv_2/Add/ReadVariableOpReadVariableOp"conv_2_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ж

conv_2/AddAddconv_2/Conv2D:output:0!conv_2/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P@]
conv_2/ReluReluconv_2/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P@i
 conv_2/spatial_dropout2d_2/ShapeShapeconv_2/Relu:activations:0*
T0*
_output_shapes
:x
.conv_2/spatial_dropout2d_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_2/spatial_dropout2d_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_2/spatial_dropout2d_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_2/spatial_dropout2d_2/strided_sliceStridedSlice)conv_2/spatial_dropout2d_2/Shape:output:07conv_2/spatial_dropout2d_2/strided_slice/stack:output:09conv_2/spatial_dropout2d_2/strided_slice/stack_1:output:09conv_2/spatial_dropout2d_2/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_2/spatial_dropout2d_2/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_2/spatial_dropout2d_2/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_2/spatial_dropout2d_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_2/spatial_dropout2d_2/strided_slice_1StridedSlice)conv_2/spatial_dropout2d_2/Shape:output:09conv_2/spatial_dropout2d_2/strided_slice_1/stack:output:0;conv_2/spatial_dropout2d_2/strided_slice_1/stack_1:output:0;conv_2/spatial_dropout2d_2/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: М
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Щ
concatConcatV2conv_2/Relu:activations:0MaxPool:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€-PА§
IdentityIdentityconv_2/Relu:activations:0^conv_2/Add/ReadVariableOp^conv_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€-P@"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::26
conv_2/Add/ReadVariableOpconv_2/Add/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
ѕ:
а

@__inference_model_layer_call_and_return_conditional_losses_12933
input_tensor/
+encode_block_statefulpartitionedcall_args_1/
+encode_block_statefulpartitionedcall_args_21
-encode_block_1_statefulpartitionedcall_args_11
-encode_block_1_statefulpartitionedcall_args_21
-encode_block_2_statefulpartitionedcall_args_11
-encode_block_2_statefulpartitionedcall_args_21
-encode_block_3_statefulpartitionedcall_args_11
-encode_block_3_statefulpartitionedcall_args_2/
+decode_block_statefulpartitionedcall_args_1/
+decode_block_statefulpartitionedcall_args_21
-decode_block_1_statefulpartitionedcall_args_11
-decode_block_1_statefulpartitionedcall_args_21
-decode_block_2_statefulpartitionedcall_args_11
-decode_block_2_statefulpartitionedcall_args_21
-decode_block_3_statefulpartitionedcall_args_11
-decode_block_3_statefulpartitionedcall_args_21
-decode_block_4_statefulpartitionedcall_args_11
-decode_block_4_statefulpartitionedcall_args_2
identityИҐ$decode_block/StatefulPartitionedCallҐ&decode_block_1/StatefulPartitionedCallҐ&decode_block_2/StatefulPartitionedCallҐ&decode_block_3/StatefulPartitionedCallҐ&decode_block_4/StatefulPartitionedCallҐ$encode_block/StatefulPartitionedCallҐ&encode_block_1/StatefulPartitionedCallҐ&encode_block_2/StatefulPartitionedCallҐ&encode_block_3/StatefulPartitionedCall™
$encode_block/StatefulPartitionedCallStatefulPartitionedCallinput_tensor+encode_block_statefulpartitionedcall_args_1+encode_block_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12305*P
fKRI
G__inference_encode_block_layer_call_and_return_conditional_losses_12291*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€іј@“
&encode_block_1/StatefulPartitionedCallStatefulPartitionedCall-encode_block/StatefulPartitionedCall:output:0-encode_block_1_statefulpartitionedcall_args_1-encode_block_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12368*R
fMRK
I__inference_encode_block_1_layer_call_and_return_conditional_losses_12354*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z†@”
&encode_block_2/StatefulPartitionedCallStatefulPartitionedCall/encode_block_1/StatefulPartitionedCall:output:0-encode_block_2_statefulpartitionedcall_args_1-encode_block_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12431*R
fMRK
I__inference_encode_block_2_layer_call_and_return_conditional_losses_12417*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P@”
&encode_block_3/StatefulPartitionedCallStatefulPartitionedCall/encode_block_2/StatefulPartitionedCall:output:0-encode_block_3_statefulpartitionedcall_args_1-encode_block_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12494*R
fMRK
I__inference_encode_block_3_layer_call_and_return_conditional_losses_12480*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€(@Ћ
$decode_block/StatefulPartitionedCallStatefulPartitionedCall/encode_block_3/StatefulPartitionedCall:output:0+decode_block_statefulpartitionedcall_args_1+decode_block_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12555*P
fKRI
G__inference_decode_block_layer_call_and_return_conditional_losses_12541*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P “
&decode_block_1/StatefulPartitionedCallStatefulPartitionedCall-decode_block/StatefulPartitionedCall:output:0-decode_block_1_statefulpartitionedcall_args_1-decode_block_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12616*R
fMRK
I__inference_decode_block_1_layer_call_and_return_conditional_losses_12602*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† ‘
&decode_block_2/StatefulPartitionedCallStatefulPartitionedCall/encode_block_2/StatefulPartitionedCall:output:0-decode_block_2_statefulpartitionedcall_args_1-decode_block_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12677*R
fMRK
I__inference_decode_block_2_layer_call_and_return_conditional_losses_12663*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† ‘
&decode_block_3/StatefulPartitionedCallStatefulPartitionedCall/encode_block_1/StatefulPartitionedCall:output:0-decode_block_3_statefulpartitionedcall_args_1-decode_block_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12738*R
fMRK
I__inference_decode_block_3_layer_call_and_return_conditional_losses_12724*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: €
concatConcatV2/decode_block_1/StatefulPartitionedCall:output:0/decode_block_2/StatefulPartitionedCall:output:0/decode_block_3/StatefulPartitionedCall:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€Z†`µ
&decode_block_4/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0-decode_block_4_statefulpartitionedcall_args_1-decode_block_4_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12789*R
fMRK
I__inference_decode_block_4_layer_call_and_return_conditional_losses_12775*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иА
SigmoidSigmoid/decode_block_4/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€иА 
IdentityIdentitySigmoid:y:0%^decode_block/StatefulPartitionedCall'^decode_block_1/StatefulPartitionedCall'^decode_block_2/StatefulPartitionedCall'^decode_block_3/StatefulPartitionedCall'^decode_block_4/StatefulPartitionedCall%^encode_block/StatefulPartitionedCall'^encode_block_1/StatefulPartitionedCall'^encode_block_2/StatefulPartitionedCall'^encode_block_3/StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::2P
&encode_block_1/StatefulPartitionedCall&encode_block_1/StatefulPartitionedCall2P
&encode_block_2/StatefulPartitionedCall&encode_block_2/StatefulPartitionedCall2P
&encode_block_3/StatefulPartitionedCall&encode_block_3/StatefulPartitionedCall2P
&decode_block_1/StatefulPartitionedCall&decode_block_1/StatefulPartitionedCall2P
&decode_block_2/StatefulPartitionedCall&decode_block_2/StatefulPartitionedCall2P
&decode_block_3/StatefulPartitionedCall&decode_block_3/StatefulPartitionedCall2L
$encode_block/StatefulPartitionedCall$encode_block/StatefulPartitionedCall2P
&decode_block_4/StatefulPartitionedCall&decode_block_4/StatefulPartitionedCall2L
$decode_block/StatefulPartitionedCall$decode_block/StatefulPartitionedCall: : : : : : : : :	 : : : : :, (
&
_user_specified_nameinput_tensor: : : : :
 
¶
j
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_13841

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ј
O
3__inference_spatial_dropout2d_6_layer_call_fn_13935

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12177*W
fRRP
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_12176*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ч
l
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_12243

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ў
ь
I__inference_decode_block_1_layer_call_and_return_conditional_losses_12586
input_tensor)
%conv_5_conv2d_readvariableop_resource&
"conv_5_add_readvariableop_resource
identityИҐconv_5/Add/ReadVariableOpҐconv_5/Conv2D/ReadVariableOpЄ
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  ≠
conv_5/Conv2DConv2Dinput_tensor$conv_5/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ¶
conv_5/Add/ReadVariableOpReadVariableOp"conv_5_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_5/AddAddconv_5/Conv2D:output:0!conv_5/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P ]
conv_5/ReluReluconv_5/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P i
 conv_5/spatial_dropout2d_5/ShapeShapeconv_5/Relu:activations:0*
T0*
_output_shapes
:x
.conv_5/spatial_dropout2d_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_5/spatial_dropout2d_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_5/spatial_dropout2d_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_5/spatial_dropout2d_5/strided_sliceStridedSlice)conv_5/spatial_dropout2d_5/Shape:output:07conv_5/spatial_dropout2d_5/strided_slice/stack:output:09conv_5/spatial_dropout2d_5/strided_slice/stack_1:output:09conv_5/spatial_dropout2d_5/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_5/spatial_dropout2d_5/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_5/spatial_dropout2d_5/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_5/spatial_dropout2d_5/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_5/spatial_dropout2d_5/strided_slice_1StridedSlice)conv_5/spatial_dropout2d_5/Shape:output:09conv_5/spatial_dropout2d_5/strided_slice_1/stack:output:0;conv_5/spatial_dropout2d_5/strided_slice_1/stack_1:output:0;conv_5/spatial_dropout2d_5/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:≠
resize/ResizeBilinearResizeBilinearconv_5/Relu:activations:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_5/Add/ReadVariableOp^conv_5/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P ::2<
conv_5/Conv2D/ReadVariableOpconv_5/Conv2D/ReadVariableOp26
conv_5/Add/ReadVariableOpconv_5/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
я
ь
I__inference_decode_block_1_layer_call_and_return_conditional_losses_12602
input_tensor)
%conv_5_conv2d_readvariableop_resource&
"conv_5_add_readvariableop_resource
identityИҐconv_5/Add/ReadVariableOpҐconv_5/Conv2D/ReadVariableOpЄ
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  ≠
conv_5/Conv2DConv2Dinput_tensor$conv_5/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ¶
conv_5/Add/ReadVariableOpReadVariableOp"conv_5_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_5/AddAddconv_5/Conv2D:output:0!conv_5/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P ]
conv_5/ReluReluconv_5/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P Д
#conv_5/spatial_dropout2d_5/IdentityIdentityconv_5/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:ј
resize/ResizeBilinearResizeBilinear,conv_5/spatial_dropout2d_5/Identity:output:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_5/Add/ReadVariableOp^conv_5/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P ::2<
conv_5/Conv2D/ReadVariableOpconv_5/Conv2D/ReadVariableOp26
conv_5/Add/ReadVariableOpconv_5/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Х
µ
.__inference_decode_block_4_layer_call_fn_13744
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12789*R
fMRK
I__inference_decode_block_4_layer_call_and_return_conditional_losses_12775*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иАМ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†`::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
≥
M
1__inference_spatial_dropout2d_layer_call_fn_13772

inputs
identity…
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-11866*U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_11865*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
д
ь
I__inference_decode_block_3_layer_call_and_return_conditional_losses_12724
input_tensor)
%conv_7_conv2d_readvariableop_resource&
"conv_7_add_readvariableop_resource
identityИҐconv_7/Add/ReadVariableOpҐconv_7/Conv2D/ReadVariableOpЄ
conv_7/Conv2D/ReadVariableOpReadVariableOp%conv_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ Ѓ
conv_7/Conv2DConv2Dinput_tensor$conv_7/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z† ¶
conv_7/Add/ReadVariableOpReadVariableOp"conv_7_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: З

conv_7/AddAddconv_7/Conv2D:output:0!conv_7/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z† ^
conv_7/ReluReluconv_7/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z† Е
#conv_7/spatial_dropout2d_7/IdentityIdentityconv_7/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€Z† \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:ј
resize/ResizeBilinearResizeBilinear,conv_7/spatial_dropout2d_7/Identity:output:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_7/Add/ReadVariableOp^conv_7/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::2<
conv_7/Conv2D/ReadVariableOpconv_7/Conv2D/ReadVariableOp26
conv_7/Add/ReadVariableOpconv_7/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
х
j
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_11865

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
÷
т
G__inference_encode_block_layer_call_and_return_conditional_losses_12274
input_tensor'
#conv_conv2d_readvariableop_resource$
 conv_add_readvariableop_resource
identityИҐconv/Add/ReadVariableOpҐconv/Conv2D/ReadVariableOpі
conv/Conv2D/ReadVariableOpReadVariableOp#conv_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@Ђ
conv/Conv2DConv2Dinput_tensor"conv/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іј@Ґ
conv/Add/ReadVariableOpReadVariableOp conv_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@В
conv/AddAddconv/Conv2D:output:0conv/Add/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€іј@[
	conv/ReluReluconv/Add:z:0*
T0*1
_output_shapes
:€€€€€€€€€іј@c
conv/spatial_dropout2d/ShapeShapeconv/Relu:activations:0*
T0*
_output_shapes
:t
*conv/spatial_dropout2d/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:v
,conv/spatial_dropout2d/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:v
,conv/spatial_dropout2d/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ƒ
$conv/spatial_dropout2d/strided_sliceStridedSlice%conv/spatial_dropout2d/Shape:output:03conv/spatial_dropout2d/strided_slice/stack:output:05conv/spatial_dropout2d/strided_slice/stack_1:output:05conv/spatial_dropout2d/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: v
,conv/spatial_dropout2d/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:x
.conv/spatial_dropout2d/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:x
.conv/spatial_dropout2d/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ћ
&conv/spatial_dropout2d/strided_slice_1StridedSlice%conv/spatial_dropout2d/Shape:output:05conv/spatial_dropout2d/strided_slice_1/stack:output:07conv/spatial_dropout2d/strided_slice_1/stack_1:output:07conv/spatial_dropout2d/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: О
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іјV
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Ш
concatConcatV2conv/Relu:activations:0MaxPool:output:0concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€іјC†
IdentityIdentityconv/Relu:activations:0^conv/Add/ReadVariableOp^conv/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€іј@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€иА::28
conv/Conv2D/ReadVariableOpconv/Conv2D/ReadVariableOp22
conv/Add/ReadVariableOpconv/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Т
µ
.__inference_decode_block_1_layer_call_fn_13599
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12606*R
fMRK
I__inference_decode_block_1_layer_call_and_return_conditional_losses_12586*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P ::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
§
h
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_11852

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ч
l
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_13818

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ј
O
3__inference_spatial_dropout2d_3_layer_call_fn_13856

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12028*W
fRRP
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_12027*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ч
р
%__inference_model_layer_call_fn_12895
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
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*,
_gradient_op_typePartitionedCall-12874*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_12873*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иАМ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : :	 : : : : :' #
!
_user_specified_name	input_1: : : : :
 
Ј
O
3__inference_spatial_dropout2d_1_layer_call_fn_13800

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-11920*W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_11919*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
С
µ
.__inference_encode_block_2_layer_call_fn_13447
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12421*R
fMRK
I__inference_encode_block_2_layer_call_and_return_conditional_losses_12400*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P@К
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€-P@"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
Ј
O
3__inference_spatial_dropout2d_1_layer_call_fn_13795

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-11907*W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_11906*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ў
ь
I__inference_decode_block_2_layer_call_and_return_conditional_losses_13628
input_tensor)
%conv_6_conv2d_readvariableop_resource&
"conv_6_add_readvariableop_resource
identityИҐconv_6/Add/ReadVariableOpҐconv_6/Conv2D/ReadVariableOpЄ
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ≠
conv_6/Conv2DConv2Dinput_tensor$conv_6/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ¶
conv_6/Add/ReadVariableOpReadVariableOp"conv_6_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_6/AddAddconv_6/Conv2D:output:0!conv_6/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P ]
conv_6/ReluReluconv_6/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P i
 conv_6/spatial_dropout2d_6/ShapeShapeconv_6/Relu:activations:0*
T0*
_output_shapes
:x
.conv_6/spatial_dropout2d_6/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_6/spatial_dropout2d_6/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_6/spatial_dropout2d_6/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_6/spatial_dropout2d_6/strided_sliceStridedSlice)conv_6/spatial_dropout2d_6/Shape:output:07conv_6/spatial_dropout2d_6/strided_slice/stack:output:09conv_6/spatial_dropout2d_6/strided_slice/stack_1:output:09conv_6/spatial_dropout2d_6/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_6/spatial_dropout2d_6/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_6/spatial_dropout2d_6/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_6/spatial_dropout2d_6/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_6/spatial_dropout2d_6/strided_slice_1StridedSlice)conv_6/spatial_dropout2d_6/Shape:output:09conv_6/spatial_dropout2d_6/strided_slice_1/stack:output:0;conv_6/spatial_dropout2d_6/strided_slice_1/stack_1:output:0;conv_6/spatial_dropout2d_6/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:≠
resize/ResizeBilinearResizeBilinearconv_6/Relu:activations:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_6/Add/ReadVariableOp^conv_6/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::2<
conv_6/Conv2D/ReadVariableOpconv_6/Conv2D/ReadVariableOp26
conv_6/Add/ReadVariableOpconv_6/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
®x
э!
__inference__traced_save_14170
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop?
;savev2_model_encode_block_conv_variable_read_readvariableopA
=savev2_model_encode_block_conv_variable_1_read_readvariableopC
?savev2_model_encode_block_1_conv_1_variable_read_readvariableopE
Asavev2_model_encode_block_1_conv_1_variable_1_read_readvariableopC
?savev2_model_encode_block_2_conv_2_variable_read_readvariableopE
Asavev2_model_encode_block_2_conv_2_variable_1_read_readvariableopC
?savev2_model_encode_block_3_conv_3_variable_read_readvariableopE
Asavev2_model_encode_block_3_conv_3_variable_1_read_readvariableopA
=savev2_model_decode_block_conv_4_variable_read_readvariableopC
?savev2_model_decode_block_conv_4_variable_1_read_readvariableopC
?savev2_model_decode_block_1_conv_5_variable_read_readvariableopE
Asavev2_model_decode_block_1_conv_5_variable_1_read_readvariableopC
?savev2_model_decode_block_2_conv_6_variable_read_readvariableopE
Asavev2_model_decode_block_2_conv_6_variable_1_read_readvariableopC
?savev2_model_decode_block_3_conv_7_variable_read_readvariableopE
Asavev2_model_decode_block_3_conv_7_variable_1_read_readvariableopC
?savev2_model_decode_block_4_conv_8_variable_read_readvariableopE
Asavev2_model_decode_block_4_conv_8_variable_1_read_readvariableopF
Bsavev2_adam_model_encode_block_conv_variable_m_read_readvariableopH
Dsavev2_adam_model_encode_block_conv_variable_m_1_read_readvariableopJ
Fsavev2_adam_model_encode_block_1_conv_1_variable_m_read_readvariableopL
Hsavev2_adam_model_encode_block_1_conv_1_variable_m_1_read_readvariableopJ
Fsavev2_adam_model_encode_block_2_conv_2_variable_m_read_readvariableopL
Hsavev2_adam_model_encode_block_2_conv_2_variable_m_1_read_readvariableopJ
Fsavev2_adam_model_encode_block_3_conv_3_variable_m_read_readvariableopL
Hsavev2_adam_model_encode_block_3_conv_3_variable_m_1_read_readvariableopH
Dsavev2_adam_model_decode_block_conv_4_variable_m_read_readvariableopJ
Fsavev2_adam_model_decode_block_conv_4_variable_m_1_read_readvariableopJ
Fsavev2_adam_model_decode_block_1_conv_5_variable_m_read_readvariableopL
Hsavev2_adam_model_decode_block_1_conv_5_variable_m_1_read_readvariableopJ
Fsavev2_adam_model_decode_block_2_conv_6_variable_m_read_readvariableopL
Hsavev2_adam_model_decode_block_2_conv_6_variable_m_1_read_readvariableopJ
Fsavev2_adam_model_decode_block_3_conv_7_variable_m_read_readvariableopL
Hsavev2_adam_model_decode_block_3_conv_7_variable_m_1_read_readvariableopJ
Fsavev2_adam_model_decode_block_4_conv_8_variable_m_read_readvariableopL
Hsavev2_adam_model_decode_block_4_conv_8_variable_m_1_read_readvariableopF
Bsavev2_adam_model_encode_block_conv_variable_v_read_readvariableopH
Dsavev2_adam_model_encode_block_conv_variable_v_1_read_readvariableopJ
Fsavev2_adam_model_encode_block_1_conv_1_variable_v_read_readvariableopL
Hsavev2_adam_model_encode_block_1_conv_1_variable_v_1_read_readvariableopJ
Fsavev2_adam_model_encode_block_2_conv_2_variable_v_read_readvariableopL
Hsavev2_adam_model_encode_block_2_conv_2_variable_v_1_read_readvariableopJ
Fsavev2_adam_model_encode_block_3_conv_3_variable_v_read_readvariableopL
Hsavev2_adam_model_encode_block_3_conv_3_variable_v_1_read_readvariableopH
Dsavev2_adam_model_decode_block_conv_4_variable_v_read_readvariableopJ
Fsavev2_adam_model_decode_block_conv_4_variable_v_1_read_readvariableopJ
Fsavev2_adam_model_decode_block_1_conv_5_variable_v_read_readvariableopL
Hsavev2_adam_model_decode_block_1_conv_5_variable_v_1_read_readvariableopJ
Fsavev2_adam_model_decode_block_2_conv_6_variable_v_read_readvariableopL
Hsavev2_adam_model_decode_block_2_conv_6_variable_v_1_read_readvariableopJ
Fsavev2_adam_model_decode_block_3_conv_7_variable_v_read_readvariableopL
Hsavev2_adam_model_decode_block_3_conv_7_variable_v_1_read_readvariableopJ
Fsavev2_adam_model_decode_block_4_conv_8_variable_v_read_readvariableopL
Hsavev2_adam_model_decode_block_4_conv_8_variable_v_1_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_5f145388b7b240df91df7adb71aad652/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: …
SaveV2/tensor_namesConst"/device:CPU:0*т
valueиBе;B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:;е
SaveV2/shape_and_slicesConst"/device:CPU:0*К
valueАB~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:;м 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop;savev2_model_encode_block_conv_variable_read_readvariableop=savev2_model_encode_block_conv_variable_1_read_readvariableop?savev2_model_encode_block_1_conv_1_variable_read_readvariableopAsavev2_model_encode_block_1_conv_1_variable_1_read_readvariableop?savev2_model_encode_block_2_conv_2_variable_read_readvariableopAsavev2_model_encode_block_2_conv_2_variable_1_read_readvariableop?savev2_model_encode_block_3_conv_3_variable_read_readvariableopAsavev2_model_encode_block_3_conv_3_variable_1_read_readvariableop=savev2_model_decode_block_conv_4_variable_read_readvariableop?savev2_model_decode_block_conv_4_variable_1_read_readvariableop?savev2_model_decode_block_1_conv_5_variable_read_readvariableopAsavev2_model_decode_block_1_conv_5_variable_1_read_readvariableop?savev2_model_decode_block_2_conv_6_variable_read_readvariableopAsavev2_model_decode_block_2_conv_6_variable_1_read_readvariableop?savev2_model_decode_block_3_conv_7_variable_read_readvariableopAsavev2_model_decode_block_3_conv_7_variable_1_read_readvariableop?savev2_model_decode_block_4_conv_8_variable_read_readvariableopAsavev2_model_decode_block_4_conv_8_variable_1_read_readvariableopBsavev2_adam_model_encode_block_conv_variable_m_read_readvariableopDsavev2_adam_model_encode_block_conv_variable_m_1_read_readvariableopFsavev2_adam_model_encode_block_1_conv_1_variable_m_read_readvariableopHsavev2_adam_model_encode_block_1_conv_1_variable_m_1_read_readvariableopFsavev2_adam_model_encode_block_2_conv_2_variable_m_read_readvariableopHsavev2_adam_model_encode_block_2_conv_2_variable_m_1_read_readvariableopFsavev2_adam_model_encode_block_3_conv_3_variable_m_read_readvariableopHsavev2_adam_model_encode_block_3_conv_3_variable_m_1_read_readvariableopDsavev2_adam_model_decode_block_conv_4_variable_m_read_readvariableopFsavev2_adam_model_decode_block_conv_4_variable_m_1_read_readvariableopFsavev2_adam_model_decode_block_1_conv_5_variable_m_read_readvariableopHsavev2_adam_model_decode_block_1_conv_5_variable_m_1_read_readvariableopFsavev2_adam_model_decode_block_2_conv_6_variable_m_read_readvariableopHsavev2_adam_model_decode_block_2_conv_6_variable_m_1_read_readvariableopFsavev2_adam_model_decode_block_3_conv_7_variable_m_read_readvariableopHsavev2_adam_model_decode_block_3_conv_7_variable_m_1_read_readvariableopFsavev2_adam_model_decode_block_4_conv_8_variable_m_read_readvariableopHsavev2_adam_model_decode_block_4_conv_8_variable_m_1_read_readvariableopBsavev2_adam_model_encode_block_conv_variable_v_read_readvariableopDsavev2_adam_model_encode_block_conv_variable_v_1_read_readvariableopFsavev2_adam_model_encode_block_1_conv_1_variable_v_read_readvariableopHsavev2_adam_model_encode_block_1_conv_1_variable_v_1_read_readvariableopFsavev2_adam_model_encode_block_2_conv_2_variable_v_read_readvariableopHsavev2_adam_model_encode_block_2_conv_2_variable_v_1_read_readvariableopFsavev2_adam_model_encode_block_3_conv_3_variable_v_read_readvariableopHsavev2_adam_model_encode_block_3_conv_3_variable_v_1_read_readvariableopDsavev2_adam_model_decode_block_conv_4_variable_v_read_readvariableopFsavev2_adam_model_decode_block_conv_4_variable_v_1_read_readvariableopFsavev2_adam_model_decode_block_1_conv_5_variable_v_read_readvariableopHsavev2_adam_model_decode_block_1_conv_5_variable_v_1_read_readvariableopFsavev2_adam_model_decode_block_2_conv_6_variable_v_read_readvariableopHsavev2_adam_model_decode_block_2_conv_6_variable_v_1_read_readvariableopFsavev2_adam_model_decode_block_3_conv_7_variable_v_read_readvariableopHsavev2_adam_model_decode_block_3_conv_7_variable_v_1_read_readvariableopFsavev2_adam_model_decode_block_4_conv_8_variable_v_read_readvariableopHsavev2_adam_model_decode_block_4_conv_8_variable_v_1_read_readvariableop"/device:CPU:0*I
dtypes?
=2;	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:√
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 є
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:Ц
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*Ђ
_input_shapesЩ
Ц: : : : : : :@:@:@@:@:@@:@:@@:@:@ : :  : :@ : :@ : :`::@:@:@@:@:@@:@:@@:@:@ : :  : :@ : :@ : :`::@:@:@@:@:@@:@:@@:@:@ : :  : :@ : :@ : :`:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: :4 :' : : :< :/ : : : :7 :& : : :. : : :6 :! : : :) : : :1 :  : : :9 :( : : :0 :# : :	 :8 :+ : :+ '
%
_user_specified_namefile_prefix:3 :" : : :; :* :% : : :2 :- : : :: :5 :$ : : :, : :
 
ѕ:
а

@__inference_model_layer_call_and_return_conditional_losses_12873
input_tensor/
+encode_block_statefulpartitionedcall_args_1/
+encode_block_statefulpartitionedcall_args_21
-encode_block_1_statefulpartitionedcall_args_11
-encode_block_1_statefulpartitionedcall_args_21
-encode_block_2_statefulpartitionedcall_args_11
-encode_block_2_statefulpartitionedcall_args_21
-encode_block_3_statefulpartitionedcall_args_11
-encode_block_3_statefulpartitionedcall_args_2/
+decode_block_statefulpartitionedcall_args_1/
+decode_block_statefulpartitionedcall_args_21
-decode_block_1_statefulpartitionedcall_args_11
-decode_block_1_statefulpartitionedcall_args_21
-decode_block_2_statefulpartitionedcall_args_11
-decode_block_2_statefulpartitionedcall_args_21
-decode_block_3_statefulpartitionedcall_args_11
-decode_block_3_statefulpartitionedcall_args_21
-decode_block_4_statefulpartitionedcall_args_11
-decode_block_4_statefulpartitionedcall_args_2
identityИҐ$decode_block/StatefulPartitionedCallҐ&decode_block_1/StatefulPartitionedCallҐ&decode_block_2/StatefulPartitionedCallҐ&decode_block_3/StatefulPartitionedCallҐ&decode_block_4/StatefulPartitionedCallҐ$encode_block/StatefulPartitionedCallҐ&encode_block_1/StatefulPartitionedCallҐ&encode_block_2/StatefulPartitionedCallҐ&encode_block_3/StatefulPartitionedCall™
$encode_block/StatefulPartitionedCallStatefulPartitionedCallinput_tensor+encode_block_statefulpartitionedcall_args_1+encode_block_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12295*P
fKRI
G__inference_encode_block_layer_call_and_return_conditional_losses_12274*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€іј@“
&encode_block_1/StatefulPartitionedCallStatefulPartitionedCall-encode_block/StatefulPartitionedCall:output:0-encode_block_1_statefulpartitionedcall_args_1-encode_block_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12358*R
fMRK
I__inference_encode_block_1_layer_call_and_return_conditional_losses_12337*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z†@”
&encode_block_2/StatefulPartitionedCallStatefulPartitionedCall/encode_block_1/StatefulPartitionedCall:output:0-encode_block_2_statefulpartitionedcall_args_1-encode_block_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12421*R
fMRK
I__inference_encode_block_2_layer_call_and_return_conditional_losses_12400*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P@”
&encode_block_3/StatefulPartitionedCallStatefulPartitionedCall/encode_block_2/StatefulPartitionedCall:output:0-encode_block_3_statefulpartitionedcall_args_1-encode_block_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12484*R
fMRK
I__inference_encode_block_3_layer_call_and_return_conditional_losses_12463*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€(@Ћ
$decode_block/StatefulPartitionedCallStatefulPartitionedCall/encode_block_3/StatefulPartitionedCall:output:0+decode_block_statefulpartitionedcall_args_1+decode_block_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12545*P
fKRI
G__inference_decode_block_layer_call_and_return_conditional_losses_12525*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P “
&decode_block_1/StatefulPartitionedCallStatefulPartitionedCall-decode_block/StatefulPartitionedCall:output:0-decode_block_1_statefulpartitionedcall_args_1-decode_block_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12606*R
fMRK
I__inference_decode_block_1_layer_call_and_return_conditional_losses_12586*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† ‘
&decode_block_2/StatefulPartitionedCallStatefulPartitionedCall/encode_block_2/StatefulPartitionedCall:output:0-decode_block_2_statefulpartitionedcall_args_1-decode_block_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12667*R
fMRK
I__inference_decode_block_2_layer_call_and_return_conditional_losses_12647*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† ‘
&decode_block_3/StatefulPartitionedCallStatefulPartitionedCall/encode_block_1/StatefulPartitionedCall:output:0-decode_block_3_statefulpartitionedcall_args_1-decode_block_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12728*R
fMRK
I__inference_decode_block_3_layer_call_and_return_conditional_losses_12708*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: €
concatConcatV2/decode_block_1/StatefulPartitionedCall:output:0/decode_block_2/StatefulPartitionedCall:output:0/decode_block_3/StatefulPartitionedCall:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€Z†`µ
&decode_block_4/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0-decode_block_4_statefulpartitionedcall_args_1-decode_block_4_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12779*R
fMRK
I__inference_decode_block_4_layer_call_and_return_conditional_losses_12761*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иА
SigmoidSigmoid/decode_block_4/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€иА 
IdentityIdentitySigmoid:y:0%^decode_block/StatefulPartitionedCall'^decode_block_1/StatefulPartitionedCall'^decode_block_2/StatefulPartitionedCall'^decode_block_3/StatefulPartitionedCall'^decode_block_4/StatefulPartitionedCall%^encode_block/StatefulPartitionedCall'^encode_block_1/StatefulPartitionedCall'^encode_block_2/StatefulPartitionedCall'^encode_block_3/StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::2P
&encode_block_1/StatefulPartitionedCall&encode_block_1/StatefulPartitionedCall2P
&encode_block_2/StatefulPartitionedCall&encode_block_2/StatefulPartitionedCall2P
&decode_block_1/StatefulPartitionedCall&decode_block_1/StatefulPartitionedCall2P
&encode_block_3/StatefulPartitionedCall&encode_block_3/StatefulPartitionedCall2P
&decode_block_2/StatefulPartitionedCall&decode_block_2/StatefulPartitionedCall2L
$encode_block/StatefulPartitionedCall$encode_block/StatefulPartitionedCall2P
&decode_block_3/StatefulPartitionedCall&decode_block_3/StatefulPartitionedCall2P
&decode_block_4/StatefulPartitionedCall&decode_block_4/StatefulPartitionedCall2L
$decode_block/StatefulPartitionedCall$decode_block/StatefulPartitionedCall: : : : : : : : :	 : : : : :, (
&
_user_specified_nameinput_tensor: : : : :
 
¶
j
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_12122

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ч
l
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_13874

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Х
µ
.__inference_decode_block_4_layer_call_fn_13737
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12779*R
fMRK
I__inference_decode_block_4_layer_call_and_return_conditional_losses_12761*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иАМ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†`::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
Ў
ь
I__inference_decode_block_2_layer_call_and_return_conditional_losses_12647
input_tensor)
%conv_6_conv2d_readvariableop_resource&
"conv_6_add_readvariableop_resource
identityИҐconv_6/Add/ReadVariableOpҐconv_6/Conv2D/ReadVariableOpЄ
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ≠
conv_6/Conv2DConv2Dinput_tensor$conv_6/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ¶
conv_6/Add/ReadVariableOpReadVariableOp"conv_6_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_6/AddAddconv_6/Conv2D:output:0!conv_6/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P ]
conv_6/ReluReluconv_6/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P i
 conv_6/spatial_dropout2d_6/ShapeShapeconv_6/Relu:activations:0*
T0*
_output_shapes
:x
.conv_6/spatial_dropout2d_6/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_6/spatial_dropout2d_6/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_6/spatial_dropout2d_6/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_6/spatial_dropout2d_6/strided_sliceStridedSlice)conv_6/spatial_dropout2d_6/Shape:output:07conv_6/spatial_dropout2d_6/strided_slice/stack:output:09conv_6/spatial_dropout2d_6/strided_slice/stack_1:output:09conv_6/spatial_dropout2d_6/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_6/spatial_dropout2d_6/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_6/spatial_dropout2d_6/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_6/spatial_dropout2d_6/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_6/spatial_dropout2d_6/strided_slice_1StridedSlice)conv_6/spatial_dropout2d_6/Shape:output:09conv_6/spatial_dropout2d_6/strided_slice_1/stack:output:0;conv_6/spatial_dropout2d_6/strided_slice_1/stack_1:output:0;conv_6/spatial_dropout2d_6/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:≠
resize/ResizeBilinearResizeBilinearconv_6/Relu:activations:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_6/Add/ReadVariableOp^conv_6/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::2<
conv_6/Conv2D/ReadVariableOpconv_6/Conv2D/ReadVariableOp26
conv_6/Add/ReadVariableOpconv_6/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
№
ь
I__inference_decode_block_3_layer_call_and_return_conditional_losses_13678
input_tensor)
%conv_7_conv2d_readvariableop_resource&
"conv_7_add_readvariableop_resource
identityИҐconv_7/Add/ReadVariableOpҐconv_7/Conv2D/ReadVariableOpЄ
conv_7/Conv2D/ReadVariableOpReadVariableOp%conv_7_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ Ѓ
conv_7/Conv2DConv2Dinput_tensor$conv_7/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z† ¶
conv_7/Add/ReadVariableOpReadVariableOp"conv_7_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: З

conv_7/AddAddconv_7/Conv2D:output:0!conv_7/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z† ^
conv_7/ReluReluconv_7/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z† i
 conv_7/spatial_dropout2d_7/ShapeShapeconv_7/Relu:activations:0*
T0*
_output_shapes
:x
.conv_7/spatial_dropout2d_7/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_7/spatial_dropout2d_7/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_7/spatial_dropout2d_7/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_7/spatial_dropout2d_7/strided_sliceStridedSlice)conv_7/spatial_dropout2d_7/Shape:output:07conv_7/spatial_dropout2d_7/strided_slice/stack:output:09conv_7/spatial_dropout2d_7/strided_slice/stack_1:output:09conv_7/spatial_dropout2d_7/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_7/spatial_dropout2d_7/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_7/spatial_dropout2d_7/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_7/spatial_dropout2d_7/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_7/spatial_dropout2d_7/strided_slice_1StridedSlice)conv_7/spatial_dropout2d_7/Shape:output:09conv_7/spatial_dropout2d_7/strided_slice_1/stack:output:0;conv_7/spatial_dropout2d_7/strided_slice_1/stack_1:output:0;conv_7/spatial_dropout2d_7/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:≠
resize/ResizeBilinearResizeBilinearconv_7/Relu:activations:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_7/Add/ReadVariableOp^conv_7/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::2<
conv_7/Conv2D/ReadVariableOpconv_7/Conv2D/ReadVariableOp26
conv_7/Add/ReadVariableOpconv_7/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
ј:
џ

@__inference_model_layer_call_and_return_conditional_losses_12836
input_1/
+encode_block_statefulpartitionedcall_args_1/
+encode_block_statefulpartitionedcall_args_21
-encode_block_1_statefulpartitionedcall_args_11
-encode_block_1_statefulpartitionedcall_args_21
-encode_block_2_statefulpartitionedcall_args_11
-encode_block_2_statefulpartitionedcall_args_21
-encode_block_3_statefulpartitionedcall_args_11
-encode_block_3_statefulpartitionedcall_args_2/
+decode_block_statefulpartitionedcall_args_1/
+decode_block_statefulpartitionedcall_args_21
-decode_block_1_statefulpartitionedcall_args_11
-decode_block_1_statefulpartitionedcall_args_21
-decode_block_2_statefulpartitionedcall_args_11
-decode_block_2_statefulpartitionedcall_args_21
-decode_block_3_statefulpartitionedcall_args_11
-decode_block_3_statefulpartitionedcall_args_21
-decode_block_4_statefulpartitionedcall_args_11
-decode_block_4_statefulpartitionedcall_args_2
identityИҐ$decode_block/StatefulPartitionedCallҐ&decode_block_1/StatefulPartitionedCallҐ&decode_block_2/StatefulPartitionedCallҐ&decode_block_3/StatefulPartitionedCallҐ&decode_block_4/StatefulPartitionedCallҐ$encode_block/StatefulPartitionedCallҐ&encode_block_1/StatefulPartitionedCallҐ&encode_block_2/StatefulPartitionedCallҐ&encode_block_3/StatefulPartitionedCall•
$encode_block/StatefulPartitionedCallStatefulPartitionedCallinput_1+encode_block_statefulpartitionedcall_args_1+encode_block_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12305*P
fKRI
G__inference_encode_block_layer_call_and_return_conditional_losses_12291*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€іј@“
&encode_block_1/StatefulPartitionedCallStatefulPartitionedCall-encode_block/StatefulPartitionedCall:output:0-encode_block_1_statefulpartitionedcall_args_1-encode_block_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12368*R
fMRK
I__inference_encode_block_1_layer_call_and_return_conditional_losses_12354*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z†@”
&encode_block_2/StatefulPartitionedCallStatefulPartitionedCall/encode_block_1/StatefulPartitionedCall:output:0-encode_block_2_statefulpartitionedcall_args_1-encode_block_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12431*R
fMRK
I__inference_encode_block_2_layer_call_and_return_conditional_losses_12417*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P@”
&encode_block_3/StatefulPartitionedCallStatefulPartitionedCall/encode_block_2/StatefulPartitionedCall:output:0-encode_block_3_statefulpartitionedcall_args_1-encode_block_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12494*R
fMRK
I__inference_encode_block_3_layer_call_and_return_conditional_losses_12480*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€(@Ћ
$decode_block/StatefulPartitionedCallStatefulPartitionedCall/encode_block_3/StatefulPartitionedCall:output:0+decode_block_statefulpartitionedcall_args_1+decode_block_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12555*P
fKRI
G__inference_decode_block_layer_call_and_return_conditional_losses_12541*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P “
&decode_block_1/StatefulPartitionedCallStatefulPartitionedCall-decode_block/StatefulPartitionedCall:output:0-decode_block_1_statefulpartitionedcall_args_1-decode_block_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12616*R
fMRK
I__inference_decode_block_1_layer_call_and_return_conditional_losses_12602*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† ‘
&decode_block_2/StatefulPartitionedCallStatefulPartitionedCall/encode_block_2/StatefulPartitionedCall:output:0-decode_block_2_statefulpartitionedcall_args_1-decode_block_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12677*R
fMRK
I__inference_decode_block_2_layer_call_and_return_conditional_losses_12663*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† ‘
&decode_block_3/StatefulPartitionedCallStatefulPartitionedCall/encode_block_1/StatefulPartitionedCall:output:0-decode_block_3_statefulpartitionedcall_args_1-decode_block_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12738*R
fMRK
I__inference_decode_block_3_layer_call_and_return_conditional_losses_12724*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: €
concatConcatV2/decode_block_1/StatefulPartitionedCall:output:0/decode_block_2/StatefulPartitionedCall:output:0/decode_block_3/StatefulPartitionedCall:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€Z†`µ
&decode_block_4/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0-decode_block_4_statefulpartitionedcall_args_1-decode_block_4_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12789*R
fMRK
I__inference_decode_block_4_layer_call_and_return_conditional_losses_12775*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иА
SigmoidSigmoid/decode_block_4/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€иА 
IdentityIdentitySigmoid:y:0%^decode_block/StatefulPartitionedCall'^decode_block_1/StatefulPartitionedCall'^decode_block_2/StatefulPartitionedCall'^decode_block_3/StatefulPartitionedCall'^decode_block_4/StatefulPartitionedCall%^encode_block/StatefulPartitionedCall'^encode_block_1/StatefulPartitionedCall'^encode_block_2/StatefulPartitionedCall'^encode_block_3/StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::2P
&encode_block_1/StatefulPartitionedCall&encode_block_1/StatefulPartitionedCall2P
&encode_block_2/StatefulPartitionedCall&encode_block_2/StatefulPartitionedCall2P
&encode_block_3/StatefulPartitionedCall&encode_block_3/StatefulPartitionedCall2P
&decode_block_1/StatefulPartitionedCall&decode_block_1/StatefulPartitionedCall2P
&decode_block_2/StatefulPartitionedCall&decode_block_2/StatefulPartitionedCall2P
&decode_block_3/StatefulPartitionedCall&decode_block_3/StatefulPartitionedCall2L
$encode_block/StatefulPartitionedCall$encode_block/StatefulPartitionedCall2P
&decode_block_4/StatefulPartitionedCall&decode_block_4/StatefulPartitionedCall2L
$decode_block/StatefulPartitionedCall$decode_block/StatefulPartitionedCall: : : : : : : : :	 : : : : :' #
!
_user_specified_name	input_1: : : : :
 
¶
j
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_12014

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ј
ь
I__inference_encode_block_2_layer_call_and_return_conditional_losses_13425
input_tensor)
%conv_2_conv2d_readvariableop_resource&
"conv_2_add_readvariableop_resource
identityИҐconv_2/Add/ReadVariableOpҐconv_2/Conv2D/ReadVariableOpЄ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@≠
conv_2/Conv2DConv2Dinput_tensor$conv_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@¶
conv_2/Add/ReadVariableOpReadVariableOp"conv_2_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ж

conv_2/AddAddconv_2/Conv2D:output:0!conv_2/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P@]
conv_2/ReluReluconv_2/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P@i
 conv_2/spatial_dropout2d_2/ShapeShapeconv_2/Relu:activations:0*
T0*
_output_shapes
:x
.conv_2/spatial_dropout2d_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_2/spatial_dropout2d_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_2/spatial_dropout2d_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_2/spatial_dropout2d_2/strided_sliceStridedSlice)conv_2/spatial_dropout2d_2/Shape:output:07conv_2/spatial_dropout2d_2/strided_slice/stack:output:09conv_2/spatial_dropout2d_2/strided_slice/stack_1:output:09conv_2/spatial_dropout2d_2/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_2/spatial_dropout2d_2/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_2/spatial_dropout2d_2/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_2/spatial_dropout2d_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_2/spatial_dropout2d_2/strided_slice_1StridedSlice)conv_2/spatial_dropout2d_2/Shape:output:09conv_2/spatial_dropout2d_2/strided_slice_1/stack:output:0;conv_2/spatial_dropout2d_2/strided_slice_1/stack_1:output:0;conv_2/spatial_dropout2d_2/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: М
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Щ
concatConcatV2conv_2/Relu:activations:0MaxPool:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€-PА§
IdentityIdentityconv_2/Relu:activations:0^conv_2/Add/ReadVariableOp^conv_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€-P@"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::26
conv_2/Add/ReadVariableOpconv_2/Add/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Р
µ
.__inference_encode_block_3_layer_call_fn_13506
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12494*R
fMRK
I__inference_encode_block_3_layer_call_and_return_conditional_losses_12480*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€(@К
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€(@"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
џ
ъ
G__inference_decode_block_layer_call_and_return_conditional_losses_13542
input_tensor)
%conv_4_conv2d_readvariableop_resource&
"conv_4_add_readvariableop_resource
identityИҐconv_4/Add/ReadVariableOpҐconv_4/Conv2D/ReadVariableOpЄ
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ≠
conv_4/Conv2DConv2Dinput_tensor$conv_4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€( ¶
conv_4/Add/ReadVariableOpReadVariableOp"conv_4_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_4/AddAddconv_4/Conv2D:output:0!conv_4/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€( ]
conv_4/ReluReluconv_4/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€( Д
#conv_4/spatial_dropout2d_4/IdentityIdentityconv_4/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€( \
resize/sizeConst*
valueB"-   P   *
dtype0*
_output_shapes
:њ
resize/ResizeBilinearResizeBilinear,conv_4/spatial_dropout2d_4/Identity:output:0resize/size:output:0*
half_pixel_centers(*
T0*/
_output_shapes
:€€€€€€€€€-P ±
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_4/Add/ReadVariableOp^conv_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€-P "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€(@::2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp26
conv_4/Add/ReadVariableOpconv_4/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Ј
O
3__inference_spatial_dropout2d_2_layer_call_fn_13828

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-11974*W
fRRP
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_11973*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ў
ь
I__inference_encode_block_3_layer_call_and_return_conditional_losses_12480
input_tensor)
%conv_3_conv2d_readvariableop_resource&
"conv_3_add_readvariableop_resource
identityИҐconv_3/Add/ReadVariableOpҐconv_3/Conv2D/ReadVariableOpЄ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@≠
conv_3/Conv2DConv2Dinput_tensor$conv_3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@¶
conv_3/Add/ReadVariableOpReadVariableOp"conv_3_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ж

conv_3/AddAddconv_3/Conv2D:output:0!conv_3/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€(@]
conv_3/ReluReluconv_3/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€(@Д
#conv_3/spatial_dropout2d_3/IdentityIdentityconv_3/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€(@М
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ђ
concatConcatV2,conv_3/spatial_dropout2d_3/Identity:output:0MaxPool:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€(АЈ
IdentityIdentity,conv_3/spatial_dropout2d_3/Identity:output:0^conv_3/Add/ReadVariableOp^conv_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€(@"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::26
conv_3/Add/ReadVariableOpconv_3/Add/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
х
j
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_13762

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
С
µ
.__inference_encode_block_2_layer_call_fn_13454
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12431*R
fMRK
I__inference_encode_block_2_layer_call_and_return_conditional_losses_12417*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:€€€€€€€€€-P@К
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€-P@"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
в
ь
I__inference_encode_block_1_layer_call_and_return_conditional_losses_12354
input_tensor)
%conv_1_conv2d_readvariableop_resource&
"conv_1_add_readvariableop_resource
identityИҐconv_1/Add/ReadVariableOpҐconv_1/Conv2D/ReadVariableOpЄ
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Ѓ
conv_1/Conv2DConv2Dinput_tensor$conv_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@¶
conv_1/Add/ReadVariableOpReadVariableOp"conv_1_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@З

conv_1/AddAddconv_1/Conv2D:output:0!conv_1/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@^
conv_1/ReluReluconv_1/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@Е
#conv_1/spatial_dropout2d_1/IdentityIdentityconv_1/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@Н
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ≠
concatConcatV2,conv_1/spatial_dropout2d_1/Identity:output:0MaxPool:output:0concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€Z†АЄ
IdentityIdentity,conv_1/spatial_dropout2d_1/Identity:output:0^conv_1/Add/ReadVariableOp^conv_1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z†@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€іј@::2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp26
conv_1/Add/ReadVariableOpconv_1/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
¶
j
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_12068

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
¶
j
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_12230

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
я
ь
I__inference_decode_block_2_layer_call_and_return_conditional_losses_13642
input_tensor)
%conv_6_conv2d_readvariableop_resource&
"conv_6_add_readvariableop_resource
identityИҐconv_6/Add/ReadVariableOpҐconv_6/Conv2D/ReadVariableOpЄ
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ≠
conv_6/Conv2DConv2Dinput_tensor$conv_6/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ¶
conv_6/Add/ReadVariableOpReadVariableOp"conv_6_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_6/AddAddconv_6/Conv2D:output:0!conv_6/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P ]
conv_6/ReluReluconv_6/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P Д
#conv_6/spatial_dropout2d_6/IdentityIdentityconv_6/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:ј
resize/ResizeBilinearResizeBilinear,conv_6/spatial_dropout2d_6/Identity:output:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_6/Add/ReadVariableOp^conv_6/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::2<
conv_6/Conv2D/ReadVariableOpconv_6/Conv2D/ReadVariableOp26
conv_6/Add/ReadVariableOpconv_6/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
¶
j
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_11960

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
У
µ
.__inference_decode_block_3_layer_call_fn_13706
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12738*R
fMRK
I__inference_decode_block_3_layer_call_and_return_conditional_losses_12724*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
ч
l
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_13930

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ѓ
т
G__inference_encode_block_layer_call_and_return_conditional_losses_13336
input_tensor'
#conv_conv2d_readvariableop_resource$
 conv_add_readvariableop_resource
identityИҐconv/Add/ReadVariableOpҐconv/Conv2D/ReadVariableOpі
conv/Conv2D/ReadVariableOpReadVariableOp#conv_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@Ђ
conv/Conv2DConv2Dinput_tensor"conv/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іј@Ґ
conv/Add/ReadVariableOpReadVariableOp conv_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@В
conv/AddAddconv/Conv2D:output:0conv/Add/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€іј@[
	conv/ReluReluconv/Add:z:0*
T0*1
_output_shapes
:€€€€€€€€€іј@А
conv/spatial_dropout2d/IdentityIdentityconv/Relu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€іј@О
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іјV
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ©
concatConcatV2(conv/spatial_dropout2d/Identity:output:0MaxPool:output:0concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€іјC±
IdentityIdentity(conv/spatial_dropout2d/Identity:output:0^conv/Add/ReadVariableOp^conv/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€іј@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€иА::28
conv/Conv2D/ReadVariableOpconv/Conv2D/ReadVariableOp22
conv/Add/ReadVariableOpconv/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
в
ь
I__inference_encode_block_1_layer_call_and_return_conditional_losses_13388
input_tensor)
%conv_1_conv2d_readvariableop_resource&
"conv_1_add_readvariableop_resource
identityИҐconv_1/Add/ReadVariableOpҐconv_1/Conv2D/ReadVariableOpЄ
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Ѓ
conv_1/Conv2DConv2Dinput_tensor$conv_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@¶
conv_1/Add/ReadVariableOpReadVariableOp"conv_1_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@З

conv_1/AddAddconv_1/Conv2D:output:0!conv_1/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@^
conv_1/ReluReluconv_1/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@Е
#conv_1/spatial_dropout2d_1/IdentityIdentityconv_1/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@Н
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ≠
concatConcatV2,conv_1/spatial_dropout2d_1/Identity:output:0MaxPool:output:0concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€Z†АЄ
IdentityIdentity,conv_1/spatial_dropout2d_1/Identity:output:0^conv_1/Add/ReadVariableOp^conv_1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z†@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€іј@::2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp26
conv_1/Add/ReadVariableOpconv_1/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Џ
ь
I__inference_encode_block_2_layer_call_and_return_conditional_losses_13440
input_tensor)
%conv_2_conv2d_readvariableop_resource&
"conv_2_add_readvariableop_resource
identityИҐconv_2/Add/ReadVariableOpҐconv_2/Conv2D/ReadVariableOpЄ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@≠
conv_2/Conv2DConv2Dinput_tensor$conv_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@¶
conv_2/Add/ReadVariableOpReadVariableOp"conv_2_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ж

conv_2/AddAddconv_2/Conv2D:output:0!conv_2/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P@]
conv_2/ReluReluconv_2/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P@Д
#conv_2/spatial_dropout2d_2/IdentityIdentityconv_2/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P@М
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ђ
concatConcatV2,conv_2/spatial_dropout2d_2/Identity:output:0MaxPool:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€-PАЈ
IdentityIdentity,conv_2/spatial_dropout2d_2/Identity:output:0^conv_2/Add/ReadVariableOp^conv_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€-P@"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::26
conv_2/Add/ReadVariableOpconv_2/Add/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
¶
j
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_13953

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ж
х
%__inference_model_layer_call_fn_13298
input_tensor"
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
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*,
_gradient_op_typePartitionedCall-12934*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_12933*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иАМ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : :	 : : : : :, (
&
_user_specified_nameinput_tensor: : : : :
 
Ў
ь
I__inference_decode_block_1_layer_call_and_return_conditional_losses_13578
input_tensor)
%conv_5_conv2d_readvariableop_resource&
"conv_5_add_readvariableop_resource
identityИҐconv_5/Add/ReadVariableOpҐconv_5/Conv2D/ReadVariableOpЄ
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  ≠
conv_5/Conv2DConv2Dinput_tensor$conv_5/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ¶
conv_5/Add/ReadVariableOpReadVariableOp"conv_5_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_5/AddAddconv_5/Conv2D:output:0!conv_5/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P ]
conv_5/ReluReluconv_5/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P i
 conv_5/spatial_dropout2d_5/ShapeShapeconv_5/Relu:activations:0*
T0*
_output_shapes
:x
.conv_5/spatial_dropout2d_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_5/spatial_dropout2d_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_5/spatial_dropout2d_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_5/spatial_dropout2d_5/strided_sliceStridedSlice)conv_5/spatial_dropout2d_5/Shape:output:07conv_5/spatial_dropout2d_5/strided_slice/stack:output:09conv_5/spatial_dropout2d_5/strided_slice/stack_1:output:09conv_5/spatial_dropout2d_5/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_5/spatial_dropout2d_5/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_5/spatial_dropout2d_5/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_5/spatial_dropout2d_5/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_5/spatial_dropout2d_5/strided_slice_1StridedSlice)conv_5/spatial_dropout2d_5/Shape:output:09conv_5/spatial_dropout2d_5/strided_slice_1/stack:output:0;conv_5/spatial_dropout2d_5/strided_slice_1/stack_1:output:0;conv_5/spatial_dropout2d_5/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:≠
resize/ResizeBilinearResizeBilinearconv_5/Relu:activations:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_5/Add/ReadVariableOp^conv_5/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P ::2<
conv_5/Conv2D/ReadVariableOpconv_5/Conv2D/ReadVariableOp26
conv_5/Add/ReadVariableOpconv_5/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
§
h
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_13757

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ч
l
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_12135

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ј
O
3__inference_spatial_dropout2d_5_layer_call_fn_13907

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-12123*W
fRRP
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_12122*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
У
µ
.__inference_decode_block_3_layer_call_fn_13699
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12728*R
fMRK
I__inference_decode_block_3_layer_call_and_return_conditional_losses_12708*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
ч
l
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_13958

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
џ
ъ
G__inference_decode_block_layer_call_and_return_conditional_losses_12541
input_tensor)
%conv_4_conv2d_readvariableop_resource&
"conv_4_add_readvariableop_resource
identityИҐconv_4/Add/ReadVariableOpҐconv_4/Conv2D/ReadVariableOpЄ
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ≠
conv_4/Conv2DConv2Dinput_tensor$conv_4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€( ¶
conv_4/Add/ReadVariableOpReadVariableOp"conv_4_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_4/AddAddconv_4/Conv2D:output:0!conv_4/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€( ]
conv_4/ReluReluconv_4/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€( Д
#conv_4/spatial_dropout2d_4/IdentityIdentityconv_4/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€( \
resize/sizeConst*
valueB"-   P   *
dtype0*
_output_shapes
:њ
resize/ResizeBilinearResizeBilinear,conv_4/spatial_dropout2d_4/Identity:output:0resize/size:output:0*
half_pixel_centers(*
T0*/
_output_shapes
:€€€€€€€€€-P ±
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_4/Add/ReadVariableOp^conv_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€-P "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€(@::2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp26
conv_4/Add/ReadVariableOpconv_4/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Ф
µ
.__inference_encode_block_1_layer_call_fn_13395
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12358*R
fMRK
I__inference_encode_block_1_layer_call_and_return_conditional_losses_12337*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z†@Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€Z†@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€іј@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
я
ь
I__inference_decode_block_1_layer_call_and_return_conditional_losses_13592
input_tensor)
%conv_5_conv2d_readvariableop_resource&
"conv_5_add_readvariableop_resource
identityИҐconv_5/Add/ReadVariableOpҐconv_5/Conv2D/ReadVariableOpЄ
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:  ≠
conv_5/Conv2DConv2Dinput_tensor$conv_5/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ¶
conv_5/Add/ReadVariableOpReadVariableOp"conv_5_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_5/AddAddconv_5/Conv2D:output:0!conv_5/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P ]
conv_5/ReluReluconv_5/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P Д
#conv_5/spatial_dropout2d_5/IdentityIdentityconv_5/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:ј
resize/ResizeBilinearResizeBilinear,conv_5/spatial_dropout2d_5/Identity:output:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_5/Add/ReadVariableOp^conv_5/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P ::2<
conv_5/Conv2D/ReadVariableOpconv_5/Conv2D/ReadVariableOp26
conv_5/Add/ReadVariableOpconv_5/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
ч
l
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_13790

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ч
l
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_13846

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ѓ
т
G__inference_encode_block_layer_call_and_return_conditional_losses_12291
input_tensor'
#conv_conv2d_readvariableop_resource$
 conv_add_readvariableop_resource
identityИҐconv/Add/ReadVariableOpҐconv/Conv2D/ReadVariableOpі
conv/Conv2D/ReadVariableOpReadVariableOp#conv_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@Ђ
conv/Conv2DConv2Dinput_tensor"conv/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іј@Ґ
conv/Add/ReadVariableOpReadVariableOp conv_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@В
conv/AddAddconv/Conv2D:output:0conv/Add/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€іј@[
	conv/ReluReluconv/Add:z:0*
T0*1
_output_shapes
:€€€€€€€€€іј@А
conv/spatial_dropout2d/IdentityIdentityconv/Relu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€іј@О
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:€€€€€€€€€іјV
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ©
concatConcatV2(conv/spatial_dropout2d/Identity:output:0MaxPool:output:0concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€іјC±
IdentityIdentity(conv/spatial_dropout2d/Identity:output:0^conv/Add/ReadVariableOp^conv/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€іј@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€иА::28
conv/Conv2D/ReadVariableOpconv/Conv2D/ReadVariableOp22
conv/Add/ReadVariableOpconv/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
«
ь
I__inference_encode_block_1_layer_call_and_return_conditional_losses_13373
input_tensor)
%conv_1_conv2d_readvariableop_resource&
"conv_1_add_readvariableop_resource
identityИҐconv_1/Add/ReadVariableOpҐconv_1/Conv2D/ReadVariableOpЄ
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Ѓ
conv_1/Conv2DConv2Dinput_tensor$conv_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@¶
conv_1/Add/ReadVariableOpReadVariableOp"conv_1_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@З

conv_1/AddAddconv_1/Conv2D:output:0!conv_1/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@^
conv_1/ReluReluconv_1/Add:z:0*
T0*0
_output_shapes
:€€€€€€€€€Z†@i
 conv_1/spatial_dropout2d_1/ShapeShapeconv_1/Relu:activations:0*
T0*
_output_shapes
:x
.conv_1/spatial_dropout2d_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:z
0conv_1/spatial_dropout2d_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0conv_1/spatial_dropout2d_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ў
(conv_1/spatial_dropout2d_1/strided_sliceStridedSlice)conv_1/spatial_dropout2d_1/Shape:output:07conv_1/spatial_dropout2d_1/strided_slice/stack:output:09conv_1/spatial_dropout2d_1/strided_slice/stack_1:output:09conv_1/spatial_dropout2d_1/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: z
0conv_1/spatial_dropout2d_1/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:|
2conv_1/spatial_dropout2d_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:|
2conv_1/spatial_dropout2d_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
*conv_1/spatial_dropout2d_1/strided_slice_1StridedSlice)conv_1/spatial_dropout2d_1/Shape:output:09conv_1/spatial_dropout2d_1/strided_slice_1/stack:output:0;conv_1/spatial_dropout2d_1/strided_slice_1/stack_1:output:0;conv_1/spatial_dropout2d_1/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Н
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Ъ
concatConcatV2conv_1/Relu:activations:0MaxPool:output:0concat/axis:output:0*
T0*
N*1
_output_shapes
:€€€€€€€€€Z†А•
IdentityIdentityconv_1/Relu:activations:0^conv_1/Add/ReadVariableOp^conv_1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z†@"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€іј@::2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp26
conv_1/Add/ReadVariableOpconv_1/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Џ
ь
I__inference_encode_block_2_layer_call_and_return_conditional_losses_12417
input_tensor)
%conv_2_conv2d_readvariableop_resource&
"conv_2_add_readvariableop_resource
identityИҐconv_2/Add/ReadVariableOpҐconv_2/Conv2D/ReadVariableOpЄ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@≠
conv_2/Conv2DConv2Dinput_tensor$conv_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@¶
conv_2/Add/ReadVariableOpReadVariableOp"conv_2_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ж

conv_2/AddAddconv_2/Conv2D:output:0!conv_2/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P@]
conv_2/ReluReluconv_2/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P@Д
#conv_2/spatial_dropout2d_2/IdentityIdentityconv_2/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P@М
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ђ
concatConcatV2,conv_2/spatial_dropout2d_2/Identity:output:0MaxPool:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€-PАЈ
IdentityIdentity,conv_2/spatial_dropout2d_2/Identity:output:0^conv_2/Add/ReadVariableOp^conv_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€-P@"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†@::26
conv_2/Add/ReadVariableOpconv_2/Add/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Ј
O
3__inference_spatial_dropout2d_2_layer_call_fn_13823

inputs
identityЋ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-11961*W
fRRP
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_11960*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Т
µ
.__inference_decode_block_1_layer_call_fn_13606
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12616*R
fMRK
I__inference_decode_block_1_layer_call_and_return_conditional_losses_12602*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P ::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
я
ь
I__inference_decode_block_2_layer_call_and_return_conditional_losses_12663
input_tensor)
%conv_6_conv2d_readvariableop_resource&
"conv_6_add_readvariableop_resource
identityИҐconv_6/Add/ReadVariableOpҐconv_6/Conv2D/ReadVariableOpЄ
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ ≠
conv_6/Conv2DConv2Dinput_tensor$conv_6/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€-P ¶
conv_6/Add/ReadVariableOpReadVariableOp"conv_6_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ж

conv_6/AddAddconv_6/Conv2D:output:0!conv_6/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€-P ]
conv_6/ReluReluconv_6/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€-P Д
#conv_6/spatial_dropout2d_6/IdentityIdentityconv_6/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€-P \
resize/sizeConst*
valueB"Z   †   *
dtype0*
_output_shapes
:ј
resize/ResizeBilinearResizeBilinear,conv_6/spatial_dropout2d_6/Identity:output:0resize/size:output:0*
half_pixel_centers(*
T0*0
_output_shapes
:€€€€€€€€€Z† ≤
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_6/Add/ReadVariableOp^conv_6/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::2<
conv_6/Conv2D/ReadVariableOpconv_6/Conv2D/ReadVariableOp26
conv_6/Add/ReadVariableOpconv_6/Add/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
а
ь
I__inference_decode_block_4_layer_call_and_return_conditional_losses_12761
input_tensor)
%conv_8_conv2d_readvariableop_resource&
"conv_8_add_readvariableop_resource
identityИҐconv_8/Add/ReadVariableOpҐconv_8/Conv2D/ReadVariableOpЄ
conv_8/Conv2D/ReadVariableOpReadVariableOp%conv_8_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:`Ѓ
conv_8/Conv2DConv2Dinput_tensor$conv_8/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Z†¶
conv_8/Add/ReadVariableOpReadVariableOp"conv_8_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:З

conv_8/AddAddconv_8/Conv2D:output:0!conv_8/Add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Z†\
resize/sizeConst*
valueB"h  А  *
dtype0*
_output_shapes
:£
resize/ResizeBilinearResizeBilinearconv_8/Add:z:0resize/size:output:0*
half_pixel_centers(*
T0*1
_output_shapes
:€€€€€€€€€иА≥
IdentityIdentity&resize/ResizeBilinear:resized_images:0^conv_8/Add/ReadVariableOp^conv_8/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€Z†`::26
conv_8/Add/ReadVariableOpconv_8/Add/ReadVariableOp2<
conv_8/Conv2D/ReadVariableOpconv_8/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
ч
р
%__inference_model_layer_call_fn_12955
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
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*,
_gradient_op_typePartitionedCall-12934*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_12933*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:€€€€€€€€€иАМ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€иА"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€иА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : :	 : : : : :' #
!
_user_specified_name	input_1: : : : :
 
ў
ь
I__inference_encode_block_3_layer_call_and_return_conditional_losses_13492
input_tensor)
%conv_3_conv2d_readvariableop_resource&
"conv_3_add_readvariableop_resource
identityИҐconv_3/Add/ReadVariableOpҐconv_3/Conv2D/ReadVariableOpЄ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@≠
conv_3/Conv2DConv2Dinput_tensor$conv_3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@¶
conv_3/Add/ReadVariableOpReadVariableOp"conv_3_add_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ж

conv_3/AddAddconv_3/Conv2D:output:0!conv_3/Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€(@]
conv_3/ReluReluconv_3/Add:z:0*
T0*/
_output_shapes
:€€€€€€€€€(@Д
#conv_3/spatial_dropout2d_3/IdentityIdentityconv_3/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€(@М
MaxPoolMaxPoolinput_tensor*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€(@V
concat/axisConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ђ
concatConcatV2,conv_3/spatial_dropout2d_3/Identity:output:0MaxPool:output:0concat/axis:output:0*
T0*
N*0
_output_shapes
:€€€€€€€€€(АЈ
IdentityIdentity,conv_3/spatial_dropout2d_3/Identity:output:0^conv_3/Add/ReadVariableOp^conv_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€(@"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::26
conv_3/Add/ReadVariableOpconv_3/Add/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp: :, (
&
_user_specified_nameinput_tensor: 
Т
µ
.__inference_decode_block_2_layer_call_fn_13649
input_tensor"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_tensorstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-12667*R
fMRK
I__inference_decode_block_2_layer_call_and_return_conditional_losses_12647*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:€€€€€€€€€Z† Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€Z† "
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€-P@::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameinput_tensor: 
¶
j
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_13925

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: _
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ў
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*њ
serving_defaultЂ
E
input_1:
serving_default_input_1:0€€€€€€€€€иАF
output_1:
StatefulPartitionedCall:0€€€€€€€€€иАtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Цн
•
	enc_0
	enc_1
	enc_2
	enc_3

dec_1a

dec_1b

dec_2a

dec_3a
	dec_out

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
ї_default_save_signature
+Љ&call_and_return_all_conditional_losses
љ__call__"о
_tf_keras_model‘{"class_name": "MODEL", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "MODEL"}, "training_config": {"loss": "binary_crossentropy", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005000000237487257, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
б
	dilations
conv_encode
	variables
trainable_variables
regularization_losses
	keras_api
+Њ&call_and_return_all_conditional_losses
њ__call__"∞
_tf_keras_layerЦ{"class_name": "encode_block", "name": "encode_block", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
г
	dilations
conv_encode
	variables
trainable_variables
regularization_losses
	keras_api
+ј&call_and_return_all_conditional_losses
Ѕ__call__"≤
_tf_keras_layerШ{"class_name": "encode_block", "name": "encode_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
г
	dilations
conv_encode
	variables
trainable_variables
 regularization_losses
!	keras_api
+¬&call_and_return_all_conditional_losses
√__call__"≤
_tf_keras_layerШ{"class_name": "encode_block", "name": "encode_block_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
г
"	dilations
#conv_encode
$	variables
%trainable_variables
&regularization_losses
'	keras_api
+ƒ&call_and_return_all_conditional_losses
≈__call__"≤
_tf_keras_layerШ{"class_name": "encode_block", "name": "encode_block_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
÷
	(shape
)conv
*	variables
+trainable_variables
,regularization_losses
-	keras_api
+∆&call_and_return_all_conditional_losses
«__call__"∞
_tf_keras_layerЦ{"class_name": "decode_block", "name": "decode_block", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
Ў
	.shape
/conv
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+»&call_and_return_all_conditional_losses
…__call__"≤
_tf_keras_layerШ{"class_name": "decode_block", "name": "decode_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
Ў
	4shape
5conv
6	variables
7trainable_variables
8regularization_losses
9	keras_api
+ &call_and_return_all_conditional_losses
Ћ__call__"≤
_tf_keras_layerШ{"class_name": "decode_block", "name": "decode_block_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
Ў
	:shape
;conv
<	variables
=trainable_variables
>regularization_losses
?	keras_api
+ћ&call_and_return_all_conditional_losses
Ќ__call__"≤
_tf_keras_layerШ{"class_name": "decode_block", "name": "decode_block_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
Ў
	@shape
Aconv
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
+ќ&call_and_return_all_conditional_losses
ѕ__call__"≤
_tf_keras_layerШ{"class_name": "decode_block", "name": "decode_block_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
ї
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_rateKmЧLmШMmЩNmЪOmЫPmЬQmЭRmЮSmЯTm†Um°VmҐWm£Xm§Ym•Zm¶[mІ\m®Kv©Lv™MvЂNvђOv≠PvЃQvѓRv∞Sv±Tv≤Uv≥VvіWvµXvґYvЈZvЄ[vє\vЇ"
	optimizer
¶
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12
X13
Y14
Z15
[16
\17"
trackable_list_wrapper
¶
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12
X13
Y14
Z15
[16
\17"
trackable_list_wrapper
 "
trackable_list_wrapper
ї
	variables
]layer_regularization_losses

^layers
trainable_variables
_metrics
regularization_losses
`non_trainable_variables
љ__call__
ї_default_save_signature
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
-
–serving_default"
signature_map
 "
trackable_list_wrapper
«
adr
Kw
Lb
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
+—&call_and_return_all_conditional_losses
“__call__"†
_tf_keras_layerЖ{"class_name": "CONV", "name": "conv", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
	variables
flayer_regularization_losses

glayers
trainable_variables
hmetrics
regularization_losses
inon_trainable_variables
њ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
…
jdr
Mw
Nb
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
+”&call_and_return_all_conditional_losses
‘__call__"Ґ
_tf_keras_layerИ{"class_name": "CONV", "name": "conv_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
	variables
olayer_regularization_losses

players
trainable_variables
qmetrics
regularization_losses
rnon_trainable_variables
Ѕ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
…
sdr
Ow
Pb
t	variables
utrainable_variables
vregularization_losses
w	keras_api
+’&call_and_return_all_conditional_losses
÷__call__"Ґ
_tf_keras_layerИ{"class_name": "CONV", "name": "conv_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
	variables
xlayer_regularization_losses

ylayers
trainable_variables
zmetrics
 regularization_losses
{non_trainable_variables
√__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 
|dr
Qw
Rb
}	variables
~trainable_variables
regularization_losses
А	keras_api
+„&call_and_return_all_conditional_losses
Ў__call__"Ґ
_tf_keras_layerИ{"class_name": "CONV", "name": "conv_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
$	variables
 Бlayer_regularization_losses
Вlayers
%trainable_variables
Гmetrics
&regularization_losses
Дnon_trainable_variables
≈__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
ќ
Еdr
Sw
Tb
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
+ў&call_and_return_all_conditional_losses
Џ__call__"Ґ
_tf_keras_layerИ{"class_name": "CONV", "name": "conv_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
*	variables
 Кlayer_regularization_losses
Лlayers
+trainable_variables
Мmetrics
,regularization_losses
Нnon_trainable_variables
«__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
ќ
Оdr
Uw
Vb
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
+џ&call_and_return_all_conditional_losses
№__call__"Ґ
_tf_keras_layerИ{"class_name": "CONV", "name": "conv_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
0	variables
 Уlayer_regularization_losses
Фlayers
1trainable_variables
Хmetrics
2regularization_losses
Цnon_trainable_variables
…__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
ќ
Чdr
Ww
Xb
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
+Ё&call_and_return_all_conditional_losses
ё__call__"Ґ
_tf_keras_layerИ{"class_name": "CONV", "name": "conv_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
6	variables
 Ьlayer_regularization_losses
Эlayers
7trainable_variables
Юmetrics
8regularization_losses
Яnon_trainable_variables
Ћ__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
ќ
†dr
Yw
Zb
°	variables
Ґtrainable_variables
£regularization_losses
§	keras_api
+я&call_and_return_all_conditional_losses
а__call__"Ґ
_tf_keras_layerИ{"class_name": "CONV", "name": "conv_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
<	variables
 •layer_regularization_losses
¶layers
=trainable_variables
Іmetrics
>regularization_losses
®non_trainable_variables
Ќ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
ќ
©dr
[w
\b
™	variables
Ђtrainable_variables
ђregularization_losses
≠	keras_api
+б&call_and_return_all_conditional_losses
в__call__"Ґ
_tf_keras_layerИ{"class_name": "CONV", "name": "conv_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
B	variables
 Ѓlayer_regularization_losses
ѓlayers
Ctrainable_variables
∞metrics
Dregularization_losses
±non_trainable_variables
ѕ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
::8@2 model/encode_block/conv/Variable
.:,@2 model/encode_block/conv/Variable
>:<@@2$model/encode_block_1/conv_1/Variable
2:0@2$model/encode_block_1/conv_1/Variable
>:<@@2$model/encode_block_2/conv_2/Variable
2:0@2$model/encode_block_2/conv_2/Variable
>:<@@2$model/encode_block_3/conv_3/Variable
2:0@2$model/encode_block_3/conv_3/Variable
<::@ 2"model/decode_block/conv_4/Variable
0:. 2"model/decode_block/conv_4/Variable
>:<  2$model/decode_block_1/conv_5/Variable
2:0 2$model/decode_block_1/conv_5/Variable
>:<@ 2$model/decode_block_2/conv_6/Variable
2:0 2$model/decode_block_2/conv_6/Variable
>:<@ 2$model/decode_block_3/conv_7/Variable
2:0 2$model/decode_block_3/conv_7/Variable
>:<`2$model/decode_block_4/conv_8/Variable
2:02$model/decode_block_4/conv_8/Variable
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ё
≤	variables
≥trainable_variables
іregularization_losses
µ	keras_api
+г&call_and_return_all_conditional_losses
д__call__"…
_tf_keras_layerѓ{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "spatial_dropout2d", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
b	variables
 ґlayer_regularization_losses
Јlayers
ctrainable_variables
Єmetrics
dregularization_losses
єnon_trainable_variables
“__call__
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
в
Ї	variables
їtrainable_variables
Љregularization_losses
љ	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"Ќ
_tf_keras_layer≥{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "spatial_dropout2d_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
k	variables
 Њlayer_regularization_losses
њlayers
ltrainable_variables
јmetrics
mregularization_losses
Ѕnon_trainable_variables
‘__call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
в
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api
+з&call_and_return_all_conditional_losses
и__call__"Ќ
_tf_keras_layer≥{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "spatial_dropout2d_2", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
t	variables
 ∆layer_regularization_losses
«layers
utrainable_variables
»metrics
vregularization_losses
…non_trainable_variables
÷__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
в
 	variables
Ћtrainable_variables
ћregularization_losses
Ќ	keras_api
+й&call_and_return_all_conditional_losses
к__call__"Ќ
_tf_keras_layer≥{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "spatial_dropout2d_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
}	variables
 ќlayer_regularization_losses
ѕlayers
~trainable_variables
–metrics
regularization_losses
—non_trainable_variables
Ў__call__
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
в
“	variables
”trainable_variables
‘regularization_losses
’	keras_api
+л&call_and_return_all_conditional_losses
м__call__"Ќ
_tf_keras_layer≥{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "spatial_dropout2d_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
§
Ж	variables
 ÷layer_regularization_losses
„layers
Зtrainable_variables
Ўmetrics
Иregularization_losses
ўnon_trainable_variables
Џ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
в
Џ	variables
џtrainable_variables
№regularization_losses
Ё	keras_api
+н&call_and_return_all_conditional_losses
о__call__"Ќ
_tf_keras_layer≥{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "spatial_dropout2d_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
§
П	variables
 ёlayer_regularization_losses
яlayers
Рtrainable_variables
аmetrics
Сregularization_losses
бnon_trainable_variables
№__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
в
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
+п&call_and_return_all_conditional_losses
р__call__"Ќ
_tf_keras_layer≥{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "spatial_dropout2d_6", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
§
Ш	variables
 жlayer_regularization_losses
зlayers
Щtrainable_variables
иmetrics
Ъregularization_losses
йnon_trainable_variables
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
в
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
+с&call_and_return_all_conditional_losses
т__call__"Ќ
_tf_keras_layer≥{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "spatial_dropout2d_7", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
§
°	variables
 оlayer_regularization_losses
пlayers
Ґtrainable_variables
рmetrics
£regularization_losses
сnon_trainable_variables
а__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
;0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
а
т	keras_api"Ќ
_tf_keras_layer≥{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "spatial_dropout2d_8", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
§
™	variables
 уlayer_regularization_losses
фlayers
Ђtrainable_variables
хmetrics
ђregularization_losses
цnon_trainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
A0"
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
§
≤	variables
 чlayer_regularization_losses
шlayers
≥trainable_variables
щmetrics
іregularization_losses
ъnon_trainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
a0"
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
§
Ї	variables
 ыlayer_regularization_losses
ьlayers
їtrainable_variables
эmetrics
Љregularization_losses
юnon_trainable_variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
j0"
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
§
¬	variables
 €layer_regularization_losses
Аlayers
√trainable_variables
Бmetrics
ƒregularization_losses
Вnon_trainable_variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
s0"
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
§
 	variables
 Гlayer_regularization_losses
Дlayers
Ћtrainable_variables
Еmetrics
ћregularization_losses
Жnon_trainable_variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
|0"
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
§
“	variables
 Зlayer_regularization_losses
Иlayers
”trainable_variables
Йmetrics
‘regularization_losses
Кnon_trainable_variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Е0"
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
§
Џ	variables
 Лlayer_regularization_losses
Мlayers
џtrainable_variables
Нmetrics
№regularization_losses
Оnon_trainable_variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
О0"
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
§
в	variables
 Пlayer_regularization_losses
Рlayers
гtrainable_variables
Сmetrics
дregularization_losses
Тnon_trainable_variables
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Ч0"
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
§
к	variables
 Уlayer_regularization_losses
Фlayers
лtrainable_variables
Хmetrics
мregularization_losses
Цnon_trainable_variables
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
†0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
(
©0"
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
?:=@2'Adam/model/encode_block/conv/Variable/m
3:1@2'Adam/model/encode_block/conv/Variable/m
C:A@@2+Adam/model/encode_block_1/conv_1/Variable/m
7:5@2+Adam/model/encode_block_1/conv_1/Variable/m
C:A@@2+Adam/model/encode_block_2/conv_2/Variable/m
7:5@2+Adam/model/encode_block_2/conv_2/Variable/m
C:A@@2+Adam/model/encode_block_3/conv_3/Variable/m
7:5@2+Adam/model/encode_block_3/conv_3/Variable/m
A:?@ 2)Adam/model/decode_block/conv_4/Variable/m
5:3 2)Adam/model/decode_block/conv_4/Variable/m
C:A  2+Adam/model/decode_block_1/conv_5/Variable/m
7:5 2+Adam/model/decode_block_1/conv_5/Variable/m
C:A@ 2+Adam/model/decode_block_2/conv_6/Variable/m
7:5 2+Adam/model/decode_block_2/conv_6/Variable/m
C:A@ 2+Adam/model/decode_block_3/conv_7/Variable/m
7:5 2+Adam/model/decode_block_3/conv_7/Variable/m
C:A`2+Adam/model/decode_block_4/conv_8/Variable/m
7:52+Adam/model/decode_block_4/conv_8/Variable/m
?:=@2'Adam/model/encode_block/conv/Variable/v
3:1@2'Adam/model/encode_block/conv/Variable/v
C:A@@2+Adam/model/encode_block_1/conv_1/Variable/v
7:5@2+Adam/model/encode_block_1/conv_1/Variable/v
C:A@@2+Adam/model/encode_block_2/conv_2/Variable/v
7:5@2+Adam/model/encode_block_2/conv_2/Variable/v
C:A@@2+Adam/model/encode_block_3/conv_3/Variable/v
7:5@2+Adam/model/encode_block_3/conv_3/Variable/v
A:?@ 2)Adam/model/decode_block/conv_4/Variable/v
5:3 2)Adam/model/decode_block/conv_4/Variable/v
C:A  2+Adam/model/decode_block_1/conv_5/Variable/v
7:5 2+Adam/model/decode_block_1/conv_5/Variable/v
C:A@ 2+Adam/model/decode_block_2/conv_6/Variable/v
7:5 2+Adam/model/decode_block_2/conv_6/Variable/v
C:A@ 2+Adam/model/decode_block_3/conv_7/Variable/v
7:5 2+Adam/model/decode_block_3/conv_7/Variable/v
C:A`2+Adam/model/decode_block_4/conv_8/Variable/v
7:52+Adam/model/decode_block_4/conv_8/Variable/v
и2е
 __inference__wrapped_model_11813ј
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *0Ґ-
+К(
input_1€€€€€€€€€иА
»2≈
@__inference_model_layer_call_and_return_conditional_losses_12800
@__inference_model_layer_call_and_return_conditional_losses_13252
@__inference_model_layer_call_and_return_conditional_losses_12836
@__inference_model_layer_call_and_return_conditional_losses_13153Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
№2ў
%__inference_model_layer_call_fn_13298
%__inference_model_layer_call_fn_12955
%__inference_model_layer_call_fn_12895
%__inference_model_layer_call_fn_13275Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
G__inference_encode_block_layer_call_and_return_conditional_losses_13321
G__inference_encode_block_layer_call_and_return_conditional_losses_13336Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ь2Щ
,__inference_encode_block_layer_call_fn_13350
,__inference_encode_block_layer_call_fn_13343Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
I__inference_encode_block_1_layer_call_and_return_conditional_losses_13388
I__inference_encode_block_1_layer_call_and_return_conditional_losses_13373Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
.__inference_encode_block_1_layer_call_fn_13395
.__inference_encode_block_1_layer_call_fn_13402Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
I__inference_encode_block_2_layer_call_and_return_conditional_losses_13425
I__inference_encode_block_2_layer_call_and_return_conditional_losses_13440Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
.__inference_encode_block_2_layer_call_fn_13454
.__inference_encode_block_2_layer_call_fn_13447Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
I__inference_encode_block_3_layer_call_and_return_conditional_losses_13477
I__inference_encode_block_3_layer_call_and_return_conditional_losses_13492Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
.__inference_encode_block_3_layer_call_fn_13506
.__inference_encode_block_3_layer_call_fn_13499Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
G__inference_decode_block_layer_call_and_return_conditional_losses_13528
G__inference_decode_block_layer_call_and_return_conditional_losses_13542Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ь2Щ
,__inference_decode_block_layer_call_fn_13549
,__inference_decode_block_layer_call_fn_13556Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
I__inference_decode_block_1_layer_call_and_return_conditional_losses_13578
I__inference_decode_block_1_layer_call_and_return_conditional_losses_13592Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
.__inference_decode_block_1_layer_call_fn_13599
.__inference_decode_block_1_layer_call_fn_13606Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
I__inference_decode_block_2_layer_call_and_return_conditional_losses_13628
I__inference_decode_block_2_layer_call_and_return_conditional_losses_13642Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
.__inference_decode_block_2_layer_call_fn_13656
.__inference_decode_block_2_layer_call_fn_13649Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
I__inference_decode_block_3_layer_call_and_return_conditional_losses_13678
I__inference_decode_block_3_layer_call_and_return_conditional_losses_13692Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
.__inference_decode_block_3_layer_call_fn_13706
.__inference_decode_block_3_layer_call_fn_13699Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
I__inference_decode_block_4_layer_call_and_return_conditional_losses_13718
I__inference_decode_block_4_layer_call_and_return_conditional_losses_13730Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
.__inference_decode_block_4_layer_call_fn_13744
.__inference_decode_block_4_layer_call_fn_13737Ї
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
2B0
#__inference_signature_wrapper_12988input_1
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ј2љЇ
±≤≠
FullArgSpec/
args'Ъ$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_13757
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_13762і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
1__inference_spatial_dropout2d_layer_call_fn_13767
1__inference_spatial_dropout2d_layer_call_fn_13772і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_13790
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_13785і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
3__inference_spatial_dropout2d_1_layer_call_fn_13800
3__inference_spatial_dropout2d_1_layer_call_fn_13795і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_13813
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_13818і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
3__inference_spatial_dropout2d_2_layer_call_fn_13823
3__inference_spatial_dropout2d_2_layer_call_fn_13828і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_13846
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_13841і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
3__inference_spatial_dropout2d_3_layer_call_fn_13856
3__inference_spatial_dropout2d_3_layer_call_fn_13851і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_13869
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_13874і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
3__inference_spatial_dropout2d_4_layer_call_fn_13884
3__inference_spatial_dropout2d_4_layer_call_fn_13879і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_13902
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_13897і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
3__inference_spatial_dropout2d_5_layer_call_fn_13912
3__inference_spatial_dropout2d_5_layer_call_fn_13907і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_13925
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_13930і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
3__inference_spatial_dropout2d_6_layer_call_fn_13935
3__inference_spatial_dropout2d_6_layer_call_fn_13940і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_13953
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_13958і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
3__inference_spatial_dropout2d_7_layer_call_fn_13963
3__inference_spatial_dropout2d_7_layer_call_fn_13968і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 Ћ
1__inference_spatial_dropout2d_layer_call_fn_13767ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ћ
1__inference_spatial_dropout2d_layer_call_fn_13772ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€х
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_13841ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ь
.__inference_decode_block_1_layer_call_fn_13599jUVAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P 
p
™ "!К€€€€€€€€€Z† ≈
G__inference_encode_block_layer_call_and_return_conditional_losses_13336zKLCҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€иА
p 
™ "/Ґ,
%К"
0€€€€€€€€€іј@
Ъ Ќ
3__inference_spatial_dropout2d_6_layer_call_fn_13940ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ќ
3__inference_spatial_dropout2d_6_layer_call_fn_13935ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€≈
I__inference_decode_block_3_layer_call_and_return_conditional_losses_13678xYZBҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†@
p
™ ".Ґ+
$К!
0€€€€€€€€€Z† 
Ъ х
N__inference_spatial_dropout2d_3_layer_call_and_return_conditional_losses_13846ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
I__inference_decode_block_3_layer_call_and_return_conditional_losses_13692xYZBҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†@
p 
™ ".Ґ+
$К!
0€€€€€€€€€Z† 
Ъ  
@__inference_model_layer_call_and_return_conditional_losses_12800ЕKLMNOPQRSTUVWXYZ[\>Ґ;
4Ґ1
+К(
input_1€€€€€€€€€иА
p
™ "/Ґ,
%К"
0€€€€€€€€€иА
Ъ ∆
I__inference_decode_block_4_layer_call_and_return_conditional_losses_13718y[\BҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†`
p
™ "/Ґ,
%К"
0€€€€€€€€€иА
Ъ ѕ
@__inference_model_layer_call_and_return_conditional_losses_13252КKLMNOPQRSTUVWXYZ[\CҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€иА
p 
™ "/Ґ,
%К"
0€€€€€€€€€иА
Ъ ∆
I__inference_decode_block_4_layer_call_and_return_conditional_losses_13730y[\BҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†`
p 
™ "/Ґ,
%К"
0€€€€€€€€€иА
Ъ Ь
.__inference_encode_block_2_layer_call_fn_13447jOPBҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†@
p
™ " К€€€€€€€€€-P@°
%__inference_model_layer_call_fn_12895xKLMNOPQRSTUVWXYZ[\>Ґ;
4Ґ1
+К(
input_1€€€€€€€€€иА
p
™ ""К€€€€€€€€€иАі
 __inference__wrapped_model_11813ПKLMNOPQRSTUVWXYZ[\:Ґ7
0Ґ-
+К(
input_1€€€€€€€€€иА
™ "=™:
8
output_1,К)
output_1€€€€€€€€€иАЬ
.__inference_encode_block_2_layer_call_fn_13454jOPBҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†@
p 
™ " К€€€€€€€€€-P@°
%__inference_model_layer_call_fn_12955xKLMNOPQRSTUVWXYZ[\>Ґ;
4Ґ1
+К(
input_1€€€€€€€€€иА
p 
™ ""К€€€€€€€€€иАу
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_13757ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ у
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_13762ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ  
@__inference_model_layer_call_and_return_conditional_losses_12836ЕKLMNOPQRSTUVWXYZ[\>Ґ;
4Ґ1
+К(
input_1€€€€€€€€€иА
p 
™ "/Ґ,
%К"
0€€€€€€€€€иА
Ъ Ќ
3__inference_spatial_dropout2d_2_layer_call_fn_13823ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ќ
3__inference_spatial_dropout2d_4_layer_call_fn_13879ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ќ
3__inference_spatial_dropout2d_4_layer_call_fn_13884ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€х
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_13874ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ х
N__inference_spatial_dropout2d_4_layer_call_and_return_conditional_losses_13869ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ х
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_13902ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ќ
3__inference_spatial_dropout2d_2_layer_call_fn_13828ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ю
.__inference_decode_block_4_layer_call_fn_13737l[\BҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†`
p
™ ""К€€€€€€€€€иАЮ
.__inference_decode_block_4_layer_call_fn_13744l[\BҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†`
p 
™ ""К€€€€€€€€€иАЅ
G__inference_decode_block_layer_call_and_return_conditional_losses_13528vSTAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€(@
p
™ "-Ґ*
#К 
0€€€€€€€€€-P 
Ъ ∆
I__inference_encode_block_1_layer_call_and_return_conditional_losses_13373yMNCҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€іј@
p
™ ".Ґ+
$К!
0€€€€€€€€€Z†@
Ъ Ѕ
G__inference_decode_block_layer_call_and_return_conditional_losses_13542vSTAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€(@
p 
™ "-Ґ*
#К 
0€€€€€€€€€-P 
Ъ Ь
.__inference_decode_block_2_layer_call_fn_13649jWXAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P@
p
™ "!К€€€€€€€€€Z† Ь
.__inference_decode_block_2_layer_call_fn_13656jWXAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P@
p 
™ "!К€€€€€€€€€Z† ¬
#__inference_signature_wrapper_12988ЪKLMNOPQRSTUVWXYZ[\EҐB
Ґ 
;™8
6
input_1+К(
input_1€€€€€€€€€иА"=™:
8
output_1,К)
output_1€€€€€€€€€иА∆
I__inference_encode_block_1_layer_call_and_return_conditional_losses_13388yMNCҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€іј@
p 
™ ".Ґ+
$К!
0€€€€€€€€€Z†@
Ъ х
N__inference_spatial_dropout2d_5_layer_call_and_return_conditional_losses_13897ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ х
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_13930ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ х
N__inference_spatial_dropout2d_6_layer_call_and_return_conditional_losses_13925ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
I__inference_encode_block_2_layer_call_and_return_conditional_losses_13425wOPBҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†@
p
™ "-Ґ*
#К 
0€€€€€€€€€-P@
Ъ Ќ
3__inference_spatial_dropout2d_7_layer_call_fn_13963ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ы
.__inference_encode_block_3_layer_call_fn_13506iQRAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P@
p 
™ " К€€€€€€€€€(@ƒ
I__inference_encode_block_2_layer_call_and_return_conditional_losses_13440wOPBҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†@
p 
™ "-Ґ*
#К 
0€€€€€€€€€-P@
Ъ Ю
.__inference_encode_block_1_layer_call_fn_13402lMNCҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€іј@
p 
™ "!К€€€€€€€€€Z†@Ќ
3__inference_spatial_dropout2d_5_layer_call_fn_13912ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ќ
3__inference_spatial_dropout2d_5_layer_call_fn_13907ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ќ
3__inference_spatial_dropout2d_7_layer_call_fn_13968ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒ
I__inference_decode_block_1_layer_call_and_return_conditional_losses_13578wUVAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P 
p
™ ".Ґ+
$К!
0€€€€€€€€€Z† 
Ъ ƒ
I__inference_decode_block_1_layer_call_and_return_conditional_losses_13592wUVAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P 
p 
™ ".Ґ+
$К!
0€€€€€€€€€Z† 
Ъ х
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_13953ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ы
.__inference_encode_block_3_layer_call_fn_13499iQRAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P@
p
™ " К€€€€€€€€€(@Ю
.__inference_encode_block_1_layer_call_fn_13395lMNCҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€іј@
p
™ "!К€€€€€€€€€Z†@х
N__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_13958ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
I__inference_decode_block_2_layer_call_and_return_conditional_losses_13628wWXAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P@
p
™ ".Ґ+
$К!
0€€€€€€€€€Z† 
Ъ Ќ
3__inference_spatial_dropout2d_3_layer_call_fn_13851ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒ
I__inference_decode_block_2_layer_call_and_return_conditional_losses_13642wWXAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P@
p 
™ ".Ґ+
$К!
0€€€€€€€€€Z† 
Ъ Ќ
3__inference_spatial_dropout2d_1_layer_call_fn_13800ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ќ
3__inference_spatial_dropout2d_3_layer_call_fn_13856ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Э
.__inference_decode_block_3_layer_call_fn_13706kYZBҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†@
p 
™ "!К€€€€€€€€€Z† √
I__inference_encode_block_3_layer_call_and_return_conditional_losses_13477vQRAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P@
p
™ "-Ґ*
#К 
0€€€€€€€€€(@
Ъ х
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_13790ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ х
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_13785ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ х
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_13813ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ь
.__inference_decode_block_1_layer_call_fn_13606jUVAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P 
p 
™ "!К€€€€€€€€€Z† √
I__inference_encode_block_3_layer_call_and_return_conditional_losses_13492vQRAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€-P@
p 
™ "-Ґ*
#К 
0€€€€€€€€€(@
Ъ ¶
%__inference_model_layer_call_fn_13275}KLMNOPQRSTUVWXYZ[\CҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€иА
p
™ ""К€€€€€€€€€иАх
N__inference_spatial_dropout2d_2_layer_call_and_return_conditional_losses_13818ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Щ
,__inference_decode_block_layer_call_fn_13549iSTAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€(@
p
™ " К€€€€€€€€€-P Щ
,__inference_decode_block_layer_call_fn_13556iSTAҐ>
7Ґ4
.К+
input_tensor€€€€€€€€€(@
p 
™ " К€€€€€€€€€-P Э
,__inference_encode_block_layer_call_fn_13343mKLCҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€иА
p
™ ""К€€€€€€€€€іј@ѕ
@__inference_model_layer_call_and_return_conditional_losses_13153КKLMNOPQRSTUVWXYZ[\CҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€иА
p
™ "/Ґ,
%К"
0€€€€€€€€€иА
Ъ Э
,__inference_encode_block_layer_call_fn_13350mKLCҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€иА
p 
™ ""К€€€€€€€€€іј@≈
G__inference_encode_block_layer_call_and_return_conditional_losses_13321zKLCҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€иА
p
™ "/Ґ,
%К"
0€€€€€€€€€іј@
Ъ ¶
%__inference_model_layer_call_fn_13298}KLMNOPQRSTUVWXYZ[\CҐ@
9Ґ6
0К-
input_tensor€€€€€€€€€иА
p 
™ ""К€€€€€€€€€иАЌ
3__inference_spatial_dropout2d_1_layer_call_fn_13795ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Э
.__inference_decode_block_3_layer_call_fn_13699kYZBҐ?
8Ґ5
/К,
input_tensor€€€€€€€€€Z†@
p
™ "!К€€€€€€€€€Z† 