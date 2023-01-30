#!/bin/bash

# First step in processing rs-fMRI data

subject=$1
subj=${subject:3:4}
sess=${subject:1:1}
motfile=prefiltered_func_data_mcf.par
mot_thresh=0.2

base_dir="/studyFolder"
script_dir="/studyFolder/scripts"
input_dir="${base_dir}/dataFolder/${sess}/${subj}/func/RS"
feat_dir="${input_dir}/${subj}_rest.feat"
template_dir="${script_dir}/design_templates"

if [ ! -e $feat_dir ]; then

	# make directory for motion files 
	if [ ! -e ${rest_dir}/mot ]; then
		mkdir ${rest_dir}/mot
	fi

echo "Running fsl_motion_outliers"
fsl_motion_outliers -i $input_dir/resting_state -o $input_dir/resting_state_fd_confound.txt --fd --thresh=$mot_thresh -p $input_dir/mot/fd_plot -v > $input_dir/resting_state_outlier_output.txt

# change the output to a column of 1s for included TRs and 0s for censored TRs
awk '{ for(i=1; i<=NF; i++) j+=$i; if (j==1) print 0; if(j==0) print 1;  j=0 }' $input_dir/resting_state_fd_confound.txt > $input_dir/mot/resting_state_fd_confound.1D

# Make a confound file (column of 1s with length equal to TRs) if no confound.1D file exists (i.e., no motion needed censoring)
# This needs to be 356 rows of 1s 
if [ ! -e $input_dir/resting_state_fd_confound.txt ]; then
	echo $subject >> $base_dir/data/mri/processed/no_confound_qa.txt
	ntimepoints="`fslnvols $input_dir/resting_state`"
	1deval -num ${ntimepoints} -expr '1' > $input_dir/mot/resting_state_fd_confound.1D
fi

# If there's a blank confound.txt file, make a new file with a column of 1s anyway.
if [ ! -s $input_dir/resting_state_fd_confound.txt ]; then
	#Store which subjects supposedly don't have any TRs sensored due to a blank confound.txt file
	echo $subject >> $base_dir/data/mri/processed/blank_confound_qa.txt
	rm -f $input_dir/mot/resting_state_fd_confound.1D
	ntimepoints="`fslnvols $rest_dir/resting_state`"
	1deval -num ${ntimepoints} -expr '1' > $input_dir/mot/resting_state_fd_confound.1D
fi
	
# edit the feat file for the subject on the fly
if [ -e $template_dir/${subj}_design_rest_fsl.fsf ]; then
	rm -f $template_dir/${subj}_design_rest_fsl.fsf 
fi
sed -e "s/SUBID/${subj}/g" -e "s/SESS/${sess}/g" \
	$template_dir/template.fsf > \
	$template_dir/${subj}_design_rest_fsl.fsf 

## Calling feat to do motion correction and registration
echo "Calling feat"
feat $script_dir/design_templates/${subj}_design_rest_fsl.fsf

## Feat does: brain extraction, computes transformation parameters (not yet applied)

## Nuisance regressor: motion
echo "Making motion files"
 1dnorm -demean $feat_dir/mc/$motfile'[0]' $rest_dir/mot/${subj}_fd_motion.1x.1D
 1dnorm -demean $feat_dir/mc/$motfile'[1]' $rest_dir/mot/${subj}_fd_motion.2x.1D
 1dnorm -demean $feat_dir/mc/$motfile'[2]' $rest_dir/mot/${subj}_fd_motion.3x.1D
 1dnorm -demean $feat_dir/mc/$motfile'[3]' $rest_dir/mot/${subj}_fd_motion.4x.1D
 1dnorm -demean $feat_dir/mc/$motfile'[4]' $rest_dir/mot/${subj}_fd_motion.5x.1D
 1dnorm -demean $feat_dir/mc/$motfile'[5]' $rest_dir/mot/${subj}_fd_motion.6x.1D

echo "Applying warps to filtered func data"
applywarp --ref=$FSLDIR/data/standard/MNI152_T1_2mm_brain --in=$feat_dir/filtered_func_data --out=$feat_dir/${subj}_mcf2standard --warp=$feat_dir/reg/example_func2standard_warp


fi
