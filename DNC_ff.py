	
import tensorflow as tf 
from tensorflow.python.ops import rnn,rnn_cell 
import numpy as np

def oneplus(x):
	return 1+tf.log(1+tf.exp(x))


class DNC:
	def __init__(self,batch_size,input_size,output_size,memory_size,memory_location_size,num_read_heads,hidden_size):
		self.batch_size=batch_size
		self.input_size=input_size
		self.output_size=output_size
		self.memory_size=memory_size
		self.memory_location_size=memory_location_size
		self.num_read_heads=num_read_heads
		#self.num_controller_layers=num_controller_layers
		self.hidden_size=hidden_size
		self.controller_input_size=self.input_size+self.memory_location_size*self.num_read_heads
		self.controller_output_size=self.output_size+self.memory_location_size*(3+self.num_read_heads)+self.num_read_heads*5+4
		
		

		#with tf.device('/device:GPU:0'):	
		self.input=tf.placeholder(tf.float64,[self.batch_size,None,self.input_size])
		self.output=tf.placeholder(tf.float64,[self.batch_size,None,self.output_size])
		self.time_steps=tf.placeholder(tf.int64)
		#dynamic constants
		self.memory=0.00000*tf.ones([self.batch_size,self.memory_size,self.memory_location_size],tf.float64)
		self.memory_input=0.00000*tf.ones([self.batch_size,self.memory_location_size*self.num_read_heads],tf.float64)
		self.weight_read=0.00000*tf.ones([self.batch_size,self.memory_size*self.num_read_heads],tf.float64)
		
		self.weight_write=0.00000*tf.ones([self.batch_size,self.memory_size],tf.float64)
		
		self.usage=tf.zeros([self.batch_size,self.memory_size],tf.float64)
		self.link_matrix=tf.zeros([self.batch_size,self.memory_size,self.memory_size],tf.float64)
		self.precendence=tf.zeros([self.batch_size,self.memory_size],tf.float64)

		self.extern_output_time=tf.zeros([self.batch_size,0,self.output_size],dtype=tf.float64)

		#static constants
		self.sequence_length=tf.ones([self.batch_size],tf.int64)
		self.psi_ones=tf.ones([self.batch_size,self.memory_size],tf.float64)
		self.index_mapper=tf.reshape(self.memory_size*tf.constant(list(range(0,self.batch_size)),dtype=tf.int32),[self.batch_size,1])
		self.identity_memory_size=tf.eye(self.memory_size,batch_shape=[self.batch_size],dtype=tf.float64)
	



		#Controller Variables
		
		#self.controller_lstm1=rnn_cell.BasicLSTMCell(self.hidden_size,state_is_tuple=False)
		"""self.controller_output_lstm=rnn_cell.BasicLSTMCell(self.controller_output_size,state_is_tuple=False)
		cell=[]
		for i in xrange(self.num_controller_layers):
			controller_lstm=rnn_cell.BasicLSTMCell(self.hidden_size,state_is_tuple=False)
			cell.append(controller_lstm)
		cell.append(self.controller_output_lstm)
	
		self.controller=tf.contrib.rnn.MultiRNNCell(cell,state_is_tuple=False)
		
		self.controller_state=self.controller.zero_state(self.batch_size,tf.float64)"""

		self.controller_layer1=tf.Variable(tf.random_normal([self.controller_input_size,self.hidden_size],dtype=tf.float64))
                self.controller_layer2=tf.Variable(tf.random_normal([self.hidden_size,self.hidden_size],dtype=tf.float64))
                self.controller_layer3=tf.Variable(tf.random_normal([self.hidden_size,self.hidden_size],dtype=tf.float64))
                self.controller_layer4=tf.Variable(tf.random_normal([self.hidden_size,self.hidden_size],dtype=tf.float64))
                self.controller_layer5=tf.Variable(tf.random_normal([self.hidden_size,self.hidden_size],dtype=tf.float64))
                self.controller_layer6=tf.Variable(tf.random_normal([self.hidden_size,self.controller_output_size],dtype=tf.float64))


                self.controller_bias1=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
                self.controller_bias2=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
                self.controller_bias3=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
                self.controller_bias4=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
                self.controller_bias5=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
                self.controller_bias6=tf.Variable(tf.random_normal([self.controller_output_size],dtype=tf.float64))
			
	

		
		#Computational Graph
		
		

		self.i=tf.constant(0,dtype=tf.int64)

		while_loop_output=tf.while_loop(self.DNC_while_condition,self.DNC_while_loop,\
		[self.i,self.input,self.memory_input,self.weight_read,self.weight_write,self.memory,self.link_matrix,self.precendence,self.usage,self.extern_output_time,\
		],\
		[self.i.get_shape(),self.input.get_shape(),tf.TensorShape([self.batch_size,None]),tf.TensorShape([self.batch_size,None]),tf.TensorShape([self.batch_size,None]),self.memory.get_shape(),\
		self.link_matrix.get_shape(),self.precendence.get_shape(),self.usage.get_shape(),tf.TensorShape([self.batch_size,None,self.output_size]),\
		])

		_,_,_,_,_,_,_,_,_,self.extern_output_time=while_loop_output

		self.cost=tf.reduce_sum(tf.square(self.extern_output_time-self.output),name='cost')
		
                
		self.optimizer=tf.train.AdamOptimizer(0.0005).minimize(self.cost)
		#opt=tf.train.AdamOptimizer(0.0001)
                #gvs=opt.compute_gradients(self.cost)
                #capped_gvs=[(grad,var) if grad is None else (tf.clip_by_value(grad,-100,100),var) for grad,var in gvs]
                #self.optimizer=opt.apply_gradients(capped_gvs)




	def Controller(self,extern_input,memory_input):
                controller_input=tf.concat([extern_input,memory_input],axis=1)
                

		"""controller_output=tf.reshape(controller_input,[self.batch_size,1,self.controller_input_size])
                controller_output,controller_state=tf.nn.dynamic_rnn(self.controller,controller_output,sequence_length=self.sequence_length,initial_state=controller_state)
                controller_output=tf.reshape(controller_output,[self.batch_size,self.controller_output_size])"""
		
		controller_output=tf.sigmoid(tf.add(tf.matmul(controller_input,self.controller_layer1),self.controller_bias1))
                controller_output=tf.sigmoid(tf.add(tf.matmul(controller_output,self.controller_layer2),self.controller_bias2))
                controller_output=tf.sigmoid(tf.add(tf.matmul(controller_output,self.controller_layer3),self.controller_bias3))
                controller_output=tf.sigmoid(tf.add(tf.matmul(controller_output,self.controller_layer4),self.controller_bias4))
                #controller_output=tf.sigmoid(tf.add(tf.matmul(controller_output,self.controller_layer5),self.controller_bias5))
                controller_output=tf.add(tf.matmul(controller_output,self.controller_layer6),self.controller_bias6)

                
                extern_output,k_r_vector,beta_r,k_w_vector,beta_w,beta_a,e_vector,w_vector,g_a,g_w,f,pi=\
                tf.split(controller_output,[self.output_size,self.memory_location_size*self.num_read_heads,1*self.num_read_heads,\
		self.memory_location_size,1,1,self.memory_location_size,self.memory_location_size,1,1,1*self.num_read_heads,\
		3*self.num_read_heads],axis=1)

                extern_output=extern_output
                k_r_vector=k_r_vector
                beta_r=oneplus(beta_r)
                k_w_vector=k_w_vector
                beta_w=oneplus(beta_w)
		beta_a=oneplus(beta_a)
                e_vector=tf.sigmoid(e_vector)
                w_vector=w_vector
                g_a=tf.sigmoid(g_a)
                g_w=tf.sigmoid(g_w)
                f=tf.sigmoid(f)
                pi=tf.nn.softmax(pi)
                return extern_output,k_r_vector,beta_r,k_w_vector,beta_w,beta_a,e_vector,w_vector,g_a,g_w,f,pi



	def Write_Addresser(self,k_w_vector,beta_w,beta_a,g_a,g_w,f,weight_write,weight_read,memory,usage):
		psi=tf.ones([self.batch_size,self.memory_size],dtype=tf.float64)
		j=tf.constant(0)
		_,_,_,psi=tf.while_loop(self.Write_Addresser_while_condition,self.Write_Addresser_while_loop,[j,f,weight_read,psi],\
			[j.get_shape(),f.get_shape(),weight_read.get_shape(),psi.get_shape()])
		usage=tf.multiply(tf.subtract(tf.add(usage,weight_write),tf.multiply(usage,weight_write)),psi)


		#Deep Mind 2016
		"""usage_sorted_val,usage_sorted_index=tf.nn.top_k(usage,self.memory_size)
		usage_sorted_val=tf.reverse(usage_sorted_val,axis=[1])
		usage_sorted_index=tf.reverse(usage_sorted_index,axis=[1])
		allocation_unsorted=(1-usage_sorted_val)*tf.cumprod(usage_sorted_val,axis=1,exclusive=True)
		mapped_usage_sorted_index=usage_sorted_index+self.index_mapper
		flat_allocation_unsorted=tf.reshape(allocation_unsorted,(-1,))
		flat_mapped_usage_sorted_index=tf.reshape(mapped_usage_sorted_index,(-1,))
		flat_container=tf.TensorArray(tf.float64,self.batch_size*self.memory_size)
		
		flat_allocation=flat_container.scatter(flat_mapped_usage_sorted_index,flat_allocation_unsorted)
		
		flat_allocation_packed=flat_allocation.stack()
		allocation=tf.reshape(flat_allocation_packed,[self.batch_size,self.memory_size])"""

		#Cambridge 2017
		allocation=tf.nn.softmax(beta_a*(1-usage))
		
		c_write=self.Addresser_Content(k_w_vector,beta_w,memory)
		
		weight_write=g_w*(tf.add(g_a*allocation,(1-g_a)*c_write))

		
		return weight_write,usage

	def Write_Addresser_while_loop(self,j,f,weight_read,psi):
		psi=tf.multiply(psi,(1-tf.reshape(f[:,j],[self.batch_size,1])*\
		weight_read[:,self.memory_size*j:self.memory_size*(j+1)]))
		j=tf.add(j,1)
		return j,f,weight_read,psi

	def Write_Addresser_while_condition(self,i,f,weight_read,psi):
		return i<tf.constant(self.num_read_heads)

	def Addresser_Content(self,k_vector,beta,memory):
                k_vector=tf.reshape(k_vector,[self.batch_size,self.memory_location_size,1])
                norm_k_vector=tf.nn.l2_normalize(k_vector,1)
                norm_memory=tf.nn.l2_normalize(memory,2)
                beta=tf.reshape(beta,[self.batch_size,1,1])
                return tf.reshape(tf.nn.softmax(beta*tf.matmul(norm_memory,norm_k_vector)),[self.batch_size,self.memory_size])

	def TML(self,weight_write,link_matrix,precendence):
		
		
		link_matrix=tf.multiply(tf.add(tf.multiply((1-tf.add(weight_write[:,None,:],weight_write[:,:,None])),link_matrix),\
		tf.matmul(tf.reshape(weight_write,[self.batch_size,self.memory_size,1]),tf.reshape(precendence,[self.batch_size,1,self.memory_size])\
		)),(1-self.identity_memory_size))


		precendence=tf.add((1-tf.reshape(tf.reduce_sum(weight_write,axis=1),[self.batch_size,1]))*precendence,weight_write)

		return link_matrix,precendence

	def TML_Addresser(self,weight_read,link_matrix):
		weight_f=tf.matmul(link_matrix,tf.reshape(weight_read,[self.batch_size,self.memory_size,1]))
		weight_b=tf.matmul(tf.reshape(weight_read,[self.batch_size,1,self.memory_size]),link_matrix)
		weight_f=tf.reshape(weight_f,[self.batch_size,self.memory_size])
		weight_b=tf.reshape(weight_b,[self.batch_size,self.memory_size])
		return weight_f, weight_b

	def Weight_Read_Creator(self,c_read,weight_f,weight_b,pi):
		pi=tf.reshape(pi,[self.batch_size,3,1])
		weight_read=pi[:,tf.constant(0),:]*weight_b+pi[:,tf.constant(1),:]*c_read+pi[:,tf.constant(2),:]*weight_f
		return weight_read
	
	
	def Read_Addresser(self,k_r_vector,beta_r,memory):
                c_read=self.Addresser_Content(k_r_vector,beta_r,memory)
                return c_read


	def Write_Head(self,e_vector,w_vector,weight_write,memory):
		e_vector=tf.reshape(e_vector,[self.batch_size,1,self.memory_location_size])
                e_matrix=1-tf.matmul(tf.reshape(weight_write,[self.batch_size,self.memory_size,1]),e_vector)
                memory=tf.multiply(memory,e_matrix)
                w_vector=tf.reshape(w_vector,[self.batch_size,1,self.memory_location_size])
                w_matrix=tf.matmul(tf.reshape(weight_write,[self.batch_size,self.memory_size,1]),w_vector)
                memory=tf.add(memory,w_matrix)

		return memory

	def Read_Head(self,weight_read,memory):
		return tf.reshape(tf.matmul(tf.reshape(weight_read,[self.batch_size,1,self.memory_size]),memory),[self.batch_size,self.memory_location_size])


	

		
		

	def DNC_while_loop(self,i,extern_input,memory_input,weight_read,weight_write,memory,link_matrix,precendence,usage,extern_output_time,\
	):
	
		controller_output=self.Controller(extern_input[:,i,:],memory_input)

                extern_output,k_r_vector,beta_r,k_w_vector,beta_w,beta_a,e_vector,w_vector,g_a,g_w,f,pi=controller_output
		
                extern_output_time=tf.concat([extern_output_time,tf.reshape(extern_output,[self.batch_size,1,self.output_size])],axis=1)
		
		write_addresser_output=self.Write_Addresser(k_w_vector,beta_w,beta_a,g_a,g_w,f,weight_write,weight_read,memory,usage)


		weight_write,usage=write_addresser_output

                link_matrix,precendence=self.TML(weight_write,link_matrix,precendence)

                

		weight_f=tf.zeros([self.batch_size,0],dtype=tf.float64)
		weight_b=tf.zeros([self.batch_size,0],dtype=tf.float64)
		j=tf.constant(0)
                _,weight_f,weight_b,_,_=tf.while_loop(self.TML_Addresser_while_condition,self.TML_Addresser_while_loop,\
		[j,weight_f,weight_b,weight_read,link_matrix],\
		[j.get_shape(),tf.TensorShape([self.batch_size,None]),tf.TensorShape([self.batch_size,None]),weight_read.get_shape(),link_matrix.get_shape()])
	
        	
		memory=self.Write_Head(e_vector,w_vector,weight_write,memory)

                
 
		c_read=tf.zeros([self.batch_size,0],dtype=tf.float64)
		j=tf.constant(0)
		_,c_read,_,_,_=tf.while_loop(self.Read_Addresser_while_condition,self.Read_Addresser_while_loop,\
		[j,c_read,k_r_vector,beta_r,memory],\
		[j.get_shape(),tf.TensorShape([self.batch_size,None]),k_r_vector.get_shape(),beta_r.get_shape(),memory.get_shape()])
                


		weight_read=tf.zeros([self.batch_size,0],dtype=tf.float64)
		j=tf.constant(0)
		_,weight_read,_,_,_,_=tf.while_loop(self.Weight_Read_Creator_while_condition,self.Weight_Read_Creator_while_loop,\
		[j,weight_read,c_read,weight_f,weight_b,pi],\
		[j.get_shape(),tf.TensorShape([self.batch_size,None]),c_read.get_shape(),weight_f.get_shape(),weight_b.get_shape(),pi.get_shape()])

		memory_input=tf.zeros([self.batch_size,0],dtype=tf.float64)
		j=tf.constant(0)
                _,memory_input,_,_=tf.while_loop(self.Read_Head_while_condition,self.Read_Head_while_loop,\
		[j,memory_input,weight_read,memory],\
		[j.get_shape(),tf.TensorShape([self.batch_size,None]),weight_read.get_shape(),memory.get_shape()])

		

		

                

		i=tf.add(i,1)

                return i,extern_input,memory_input,weight_read,weight_write,memory,link_matrix,precendence,usage,extern_output_time


	def DNC_while_condition(self,i,extern_input,memory_input,weight_read,weight_write,memory,link_matrix,precendence,usage,extern_output_time,\
	):
		return tf.less(i,self.time_steps)


	def TML_Addresser_while_loop(self,j,weight_f,weight_b,weight_read,link_matrix):
		weight_f_,weight_b_=self.TML_Addresser(weight_read[:,self.memory_size*j:self.memory_size*(j+1)],link_matrix)
		weight_f=tf.concat([weight_f,weight_f_],axis=1)
		weight_b=tf.concat([weight_b,weight_b_],axis=1)
		j=tf.add(j,1)
		return j,weight_f,weight_b,weight_read,link_matrix

	def TML_Addresser_while_condition(self,j,weight_f,weight_b,weight_read,link_matrix):
		return j<tf.constant(self.num_read_heads)


	def Read_Addresser_while_loop(self,j,c_read,k_r_vector,beta_r,memory):
		c_read_=self.Read_Addresser(k_r_vector[:,self.memory_location_size*j:self.memory_location_size*(j+1)],\
		tf.reshape(beta_r[:,j],[self.batch_size,1]),memory)
		c_read=tf.concat([c_read,c_read_],axis=1)
		j=tf.add(j,1)
		return j,c_read,k_r_vector,beta_r,memory

	def Read_Addresser_while_condition(self,j,c_read,k_r_vector,beta_r,memory):
		return j<tf.constant(self.num_read_heads)

	def Weight_Read_Creator_while_loop(self,j,weight_read,c_read,weight_f,weight_b,pi):
		weight_read_=self.Weight_Read_Creator(c_read[:,self.memory_size*j:self.memory_size*(j+1)],\
		weight_f[:,self.memory_size*j:self.memory_size*(j+1)],\
		weight_b[:,self.memory_size*j:self.memory_size*(j+1)],\
		pi[:,3*j:3*(j+1)])
		weight_read=tf.concat([weight_read,weight_read_],axis=1)
		j=tf.add(j,1)
		return j,weight_read,c_read,weight_f,weight_b,pi

	def Weight_Read_Creator_while_condition(self,j,weight_read,c_read,weight_f,weight_b,pi):
		return j<tf.constant(self.num_read_heads)

	def Read_Head_while_loop(self,j,memory_input,weight_read,memory):
		memory_input_=self.Read_Head(weight_read[:,self.memory_size*j:self.memory_size*(j+1)],memory)
		memory_input=tf.concat([memory_input,memory_input_],axis=1)
		j=tf.add(j,1)
		return j,memory_input,weight_read,memory

	def Read_Head_while_condition(self,j,memory_input,weight_read,memory):
		return j<tf.constant(self.num_read_heads)

	def train(self,train_steps,path_to_tb_dir):
                saver=tf.train.Saver()

                sess=tf.Session()
	#	writer=tf.summary.FileWriter(path_to_tb_dir,sess.graph)
                sess.run(tf.global_variables_initializer())

                for i in xrange(train_steps):
			time_steps=10
			#time_steps=np.random.randint(low=10,high=25)
                        batch_x=np.empty((0,time_steps,self.input_size),dtype="float64")
                        batch_y=np.empty((0,time_steps,self.output_size),dtype="float64")

                        for j in xrange(self.batch_size):
                                random_vector=np.random.rand(time_steps-5,self.output_size).astype('float64')

                                batch_x=np.append(batch_x,np.append(random_vector,np.zeros((5,self.output_size)),axis=0).reshape(1,time_steps,self.input_size),axis=0)
                                batch_y=np.append(batch_y,np.append(np.zeros((5,self.output_size)),random_vector,axis=0).reshape(1,time_steps,self.output_size),axis=0)
                                if j==0:
                                        print batch_x

                        _,c,pre=sess.run([self.optimizer,self.cost,self.extern_output_time],feed_dict={self.input:batch_x,self.output:batch_y,self.time_steps:time_steps})
                        print pre[0]
			print pre[0]-batch_y[0]
                        print i,"th ",c 
	#	writer.close()
			
                saver.save(sess,"/root/python_programs/ntm")






DNC=DNC(batch_size=200,input_size=1,output_size=1,memory_size=6,memory_location_size=5,num_read_heads=3,hidden_size=100)

DNC.train(15000,'/root/python_programs/tb_dir')


